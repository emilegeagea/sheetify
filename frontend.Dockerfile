# FROM python:3.10.6-slim
#
# RUN mkdir /code
# WORKDIR /code
#
# COPY requirements_frontend.txt /code/requirements.txt
#
# RUN pip install --upgrade pip
# RUN --mount=type=cache,target=/root/.cache/pip \
    # pip install -r requirements.txt

# FROM sheetify-frontend-base:mac
FROM sheetify-frontend-base:linux

# Necessary for using the opencv-python package to create images
RUN apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev


COPY models /code/models
COPY sheets /code/sheets
COPY sheetify-website/app.py /code/app.py
CMD python3 -m streamlit run app.py --server.port $PORT
