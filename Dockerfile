FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

WORKDIR /workspace

COPY . /workspace

RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y \
    libglib2.0-0 \
    libsm6 libxext6 libxrender-dev 

# CMD ["main.py"]
# ENTRYPOINT ["python"]

