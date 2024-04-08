import os
import sys
import logging
import traceback
import uuid

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import boto3
from PIL import Image

from model import VTryOnModel
from config.config import opt
import preprocessor


logging.basicConfig(filename='error.log', level=logging.DEBUG)
app = FastAPI()
vton = VTryOnModel(opt.device)

async def generate_img(cloth_path, person_path, job_id):
    edge_path = opt.edge_path + job_id +"_edge.jpg"
    preprocessor.generate_edge(edge_exist=False, device='cpu', img_path=cloth_path, output_path=opt.edge_path,output_name=job_id + "_edge.jpg")

    preprocessor.resize(cloth_path)
    preprocessor.resize(person_path) 

    c_tensor, e_tensor, p_tensor = preprocessor.img_to_tensor(cloth_path, edge_path, person_path)
    vton.infer(c_tensor, e_tensor, p_tensor, job_id)


@app.post("/upload-images")
async def upload_image(background_tasks: BackgroundTasks, cloth: UploadFile = File(...), person: UploadFile = File(...)):
    try:
        job_id = f"{uuid.uuid4()}"
        cloth_path = opt.cloth_path + job_id + "_cloth.jpg"
        person_path = opt.person_path + job_id + "_person.jpg"
        edge_path = opt.edge_path + job_id + "_edge.jpg"
        
        with open(cloth_path, "wb+") as img_file:
            img_file.write(cloth.file.read())
        with open(person_path, "wb+") as img_file:
            img_file.write(person.file.read())

        background_tasks.add_task(generate_img, cloth_path, person_path, job_id)

        s3 = boto3.client(service_name = opt.service_name, endpoint_url=opt.endpoint_url, aws_access_key_id=opt.access_key, aws_secret_access_key=opt.secret_key)
        bucket_name = "vton-storage"
        object_name = job_id
        s3.put_object(Bucket=bucket_name, Key=object_name)
        s3.upload_file(cloth_path, bucket_name, object_name)


        return {"JOB_ID" : job_id,
                "CLOTH_IMAGE" : cloth_path,
                "PERSON_IMAGE" : person_path,
                "EDGE_IMAGE" :  edge_path,
                "RESULT_IMAGE" : opt.root + "results/" + job_id + "_result.jpg"}

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=3000)
