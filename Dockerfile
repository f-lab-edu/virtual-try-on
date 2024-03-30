FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

WORKDIR /workspace

RUN pip install fastapi
RUN pip install uvicorn==0.16.0 
# uvicorn > 0.17 에서 ayncio.run() 지원안됨(python<3.7)

RUN pip install opencv-contrib-python==3.4.1.15
RUN apt-get update -y
RUN apt-get install libglib2.0-0 -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install cupy-cuda100
RUN pip install gdown
#u2segmenation checkpoint downloading

RUN pip install python-multipart

# ARG DISABLE_CACHE
# 소스코드 변경점이 캐싱때문에 반영이 안됨
# 이 부분 이후로 캐시 비활성화
# docker build --build-arg DISABLE_CACHE=$CUR_TIME -t test:v0 .
COPY . /workspace

