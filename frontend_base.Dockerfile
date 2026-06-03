FROM python:3.10.6-slim

RUN apt update
RUN apt install -y --no-install-recommends lilypond

RUN mkdir /code
WORKDIR /code

COPY requirements_frontend.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
