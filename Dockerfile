FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

WORKDIR /worksapce

# RUN conda install cudatoolkit=9.0 -c pytorch
# RUN pip install -U cupy >> didn't work
# run pip install cupy-cuda100 >> didn't work >> CuPy is not correctly installed
RUN conda install -c conda-forge opencv

COPY . /worksapce