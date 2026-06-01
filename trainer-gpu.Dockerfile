FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS builder

RUN mkdir /code
WORKDIR /code

# General utils
RUN apt update
RUN apt install -y --no-install-recommends make git

RUN apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    curl \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Build python venv
ENV VIRTUAL_ENV=/code/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

COPY requirements_trainer_gpu.txt /code/requirements.txt

# 2. Mount the virtual environment directory as a BuildKit cache mount
RUN pip install --upgrade pip

RUN --mount=type=cache,target=${VIRTUAL_ENV} \
    --mount=type=cache,target=/root/.cache/pip \
    pip install requests && \
    # CRITICAL: Copy the installed files back into the image layer
    cp -r ${VIRTUAL_ENV} /opt/venv_build



FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
# Necessary flags for Nvidia CUDA on Google Cloud
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/


RUN mkdir /code
WORKDIR /code


# General utils
RUN apt update
RUN apt install -y --no-install-recommends make git

RUN apt install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    curl \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*


# Google Cloud utils
RUN apt-get install -y ca-certificates gnupg curl
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get install -y google-cloud-cli


# Sheetify
ENV VIRTUAL_ENV=/code/venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Get pre-built venv
COPY --from=builder /opt/venv_build ${VIRTUAL_ENV}

# COPY requirements_trainer_gpu.txt /code/requirements.txt
# RUN pip install --upgrade pip
# RUN --mount=type=cache,target=/root/.cache/pip \
    # pip install -r requirements.txt

COPY sheets /code/sheets
COPY Makefile /code/Makefile

RUN apt install -y unzip

# Mount maestro data folder to /code/data via Cloud Storage FUSE
CMD ["make", "unzip_and_train"]
