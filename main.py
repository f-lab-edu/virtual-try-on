import os
import sys
import logging
import traceback
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from PIL import Image

from model import VTryOnModel
from config.config import Configuration
import preprocess


config = Configuration()

logging.basicConfig(filename='error.log', level=logging.DEBUG)

app = FastAPI()


@app.post("/upload/")
async def upload_image(cloth: UploadFile = File(...), person: UploadFile = File(...)):
    try:
        # Model Initialization
        vton = VTryOnModel(config.device)

        # Preprocessing
        cloth_path = config.cloth_path + f"{uuid.uuid4()}.jpg"
        with open(cloth_path, "wb") as img_file:
            img_file.write(await cloth.read())

        if not config.edge_exist:
            edge_name = f"{uuid.uuid4()}.jpg"
            edge_path = config.edge_path + edge_name
            preprocess.generate_edge(edge_exist=False, img_path=cloth_path, output_name=edge_name)
        else:
            edge_path = "dataset/service_edge/ada6385c-a2e7-44a1-a48d-6b0b64d47963.jpg"

        person_path = config.person_path + f"{uuid.uuid4()}.jpg"
        with open(person_path, "wb") as img_file:
            img_file.write(await person.read())

        for img_path in [cloth_path, edge_path, person_path]:
            im = Image.open(img_path)
            if im.size != (192, 256):
                preprocess.resize(img_path)

        c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(cloth_path, edge_path, person_path)
        
        # Inference
        vton.infer(c_tensor, e_tensor, p_tensor)
        result_image_path = "results/output.jpg"
        
        return FileResponse(result_image_path, media_type="image/jpeg")

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port=8000)
