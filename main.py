import os
import sys
import logging
import traceback
import uuid

# from async_generator import asynccontextmanager
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

async def preprocessing(cloth_path, edge_path, person_path):
    for path in [cloth_path, edge_path, person_path]:
        im = Image.open(path)
        if im.size != (192, 256):
            preprocessor.resize(path)

    c_tensor, e_tensor, p_tensor = preprocessor.img_to_tensor(cloth_path, edge_path, person_path)

    return c_tensor, e_tensor, p_tensor


async def inference(c_tensor, e_tensor, p_tensor):
    vton.infer(c_tensor, e_tensor, p_tensor)
    result_image_path = opt.root + "results/output.jpg"
    
    return FileResponse(result_image_path, media_type="image/jpeg")


@app.post("/upload-images")
async def upload_image(cloth: UploadFile = File(...), person: UploadFile = File(...) ):
    try:
        cloth_path = opt.cloth_path + f"{uuid.uuid4()}.jpg"
        person_path = opt.person_path + f"{uuid.uuid4()}.jpg"
        
        with open(cloth_path, "wb+") as img_file:
            img_file.write(cloth.file.read())
        with open(person_path, "wb+") as img_file:
            img_file.write(person.file.read())

        if not opt.edge_exist:
            edge_name = f"{uuid.uuid4()}.jpg"
            edge_path = opt.edge_path + edge_name
            preprocessor.generate_edge(edge_exist=False, device='cpu', img_path=cloth_path, output_path=opt.edge_path,output_name=edge_name)
        else:
            edge_path = opt.root + "dataset/service_edge/ada6385c-a2e7-44a1-a48d-6b0b64d47963.jpg"

        c_tensor, e_tensor, p_tensor = await preprocessing(cloth_path, edge_path, person_path)
        
        await inference(c_tensor, e_tensor, p_tensor)

        result_image_path = opt.root + "results/output.jpg"
        return FileResponse(result_image_path, media_type="image/jpeg") 

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port=3000)
