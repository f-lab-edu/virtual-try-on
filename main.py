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

s3 = boto3.client(service_name = opt.service_name, endpoint_url=opt.endpoint_url, aws_access_key_id=opt.access_key, aws_secret_access_key=opt.secret_key)
bucket_name = "vton-storage"

async def generate_img(cloth_path, person_path, output_path, job_id):
    edge_path = opt.edge_path + job_id +"_edge.jpg"
    preprocessor.generate_edge(edge_exist=False, device='cpu', img_path=cloth_path, output_path=opt.edge_path,output_name=job_id + "_edge.jpg")

    preprocessor.resize(cloth_path)
    preprocessor.resize(person_path) 

    c_tensor, e_tensor, p_tensor = preprocessor.img_to_tensor(cloth_path, edge_path, person_path)
    vton.infer(c_tensor, e_tensor, p_tensor, job_id)
    
    #Upload images to NCP Object Storage
    object_name = job_id + "/"
    s3.put_object(Bucket="vton-storage", Key=object_name)
    
    s3.upload_file(cloth_path, bucket_name, object_name + cloth_path.split("/")[-1]) 
    s3.upload_file(edge_path, bucket_name, object_name + edge_path.split("/")[-1])
    s3.upload_file(person_path, bucket_name, object_name + person_path.split("/")[-1])
    s3.upload_file(output_path, bucket_name, object_name + output_path.split("/")[-1])


@app.post("/upload-images")
async def upload_image(background_tasks: BackgroundTasks, cloth: UploadFile = File(...), person: UploadFile = File(...)):
    try:
        job_id = f"{uuid.uuid4()}"
        cloth_path = opt.cloth_path + job_id + "_cloth.jpg" 
        person_path = opt.person_path + job_id + "_person.jpg"
        edge_path = opt.edge_path + job_id + "_edge.jpg"
        output_path = opt.output_path + job_id + "_output.jpg"

        with open(cloth_path, "wb+") as img_file:
            img_file.write(cloth.file.read())
        with open(person_path, "wb+") as img_file:
            img_file.write(person.file.read())

        background_tasks.add_task(generate_img, cloth_path, person_path, output_path, job_id)

        return {"JOB_ID" : job_id}

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=3000)
