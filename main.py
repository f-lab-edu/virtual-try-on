import os
import sys
import logging
import traceback
import uuid

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from PIL import Image

from model import VTryOnModel
from config.config import opt
import preprocessor


logging.basicConfig(filename='error.log', level=logging.DEBUG)
app = FastAPI()
vton = VTryOnModel(opt.device)

async def generate_img(cloth_path, person_path):
    edge_name = f"{uuid.uuid4()}.jpg"
    edge_path = opt.edge_path + edge_name
    preprocessor.generate_edge(edge_exist=False, device='cpu', img_path=cloth_path, output_path=opt.edge_path,output_name=edge_name)

    preprocessor.resize(cloth_path)
    preprocessor.resize(person_path) 

    c_tensor, e_tensor, p_tensor = preprocessor.img_to_tensor(cloth_path, edge_path, person_path)
    vton.infer(c_tensor, e_tensor, p_tensor)


@app.post("/upload-images")
async def upload_image(background_tasks: BackgroundTasks, cloth: UploadFile = File(...), person: UploadFile = File(...)):
    try:
        cloth_path = opt.cloth_path + f"{uuid.uuid4()}.jpg"
        person_path = opt.person_path + f"{uuid.uuid4()}.jpg"
        
        with open(cloth_path, "wb+") as img_file:
            img_file.write(cloth.file.read())
        with open(person_path, "wb+") as img_file:
            img_file.write(person.file.read())

        background_tasks.add_task(generate_img, cloth_path, person_path)
        return {"message": "Generating image in the background"}

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=3000)
