FROM python:3.10.6-slim

RUN mkdir /code
WORKDIR /code

RUN apt update
RUN apt install -y --no-install-recommends make git


COPY requirements_trainer.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY sheets /code/sheets


RUN apt-get install -y ca-certificates gnupg curl
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get install -y google-cloud-cli

COPY Makefile /code/Makefile

RUN apt install -y unzip

CMD ["make", "unzip_and_train"]


# docker run -v ~/.config/gcloud:/root/.config/gcloud \
          #  -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
          #  your-image-name
