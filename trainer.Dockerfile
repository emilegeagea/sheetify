FROM python:3.10.6-slim

RUN mkdir /code
WORKDIR /code

RUN apt update && apt install -y --no-install-recommends make

COPY requirements-trainer.txt /code/requirements-trainer.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-trainer.txt

COPY sheets /code/sheets

#RUN make reset_local_files

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/

#CMD ["make", "run_workflow"]
