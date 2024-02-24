import os
import sys
import logging
import traceback
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from model import VTryOnModel
import yaml
# import config
import preprocess

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = "gpu" if config["device"] else "cpu"
edge_exist = config["data"]["edge_exist"]
resize = config["data"]["resize"]
img_root = config["data"]["root"]

logging.basicConfig(filename='error.log', level=logging.DEBUG)

app = FastAPI()

@app.post("/upload/")
async def upload_image(cloth: UploadFile = File(...), person: UploadFile = File(...)):
    try:
        vton = VTryOnModel(device)

        cloth_name = f"{uuid.uuid4()}.jpg"
        cloth_path = img_root + "service_cloth/" + cloth_name
        with open(cloth_path, "wb") as img_file:
            img_file.write(await cloth.read())

        if not edge_exist:
            edge_name = f"{uuid.uuid4()}.jpg"
            edge_path = img_root + "service_edge/" + edge_name
            preprocess.generate_edge(edge_exist=False, img_path=cloth_path, output_name=edge_name)
        else:
            edge_path = img_root + "service_edge/" + "9f3fdde1-28b8-4772-ba4a-fb7220e44aa1.jpg"

        person_name = f"{uuid.uuid4()}.jpg"
        person_path = img_root + "service_img/" + person_name
        with open(person_path, "wb") as img_file:
            img_file.write(await person.read())

        preprocess.resize(resize)
        c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(cloth_path, edge_path, person_path)

        vton.infer(c_tensor, e_tensor, p_tensor)

        result_image_path = "results/output.jpg"
        
        return FileResponse(result_image_path, media_type="image/jpeg")

    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port=8000)



# if __name__ == "__main__":
#     logging.basicConfig( level=logging.DEBUG)
#     # env = sys.argv[1] if len(sys.argv) > 2 else 'dev'
#     with open("config.yaml", "r") as f:
#         config = yaml.safe_load(f)
#     # if env == 'dev':
#     #     config = config.DevelopmentConfig()
#     # elif env == 'test':
#     #     config = config.TestConfig()

#     device = "gpu" if config["device"] else "cpu"
#     edge_exist = config["data"]["edge_exist"]
#     resize = config["data"]["resize"]
#     img_root = config["data"]["root"]

        
#     vton = VTryOnModel(device)

#     preprocess.generate_edge(edge_exist)
#     preprocess.resize(resize)
#     c_tensor, e_tensor, p_tensor = preprocess.img_to_tensor(img_root)

#     vton.infer(c_tensor, e_tensor, p_tensor)


