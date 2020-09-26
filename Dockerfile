FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

#TODO requirements file with pinned versions
RUN pip install nvidia-pyindex
RUN pip install tritonclient[http,grpc]
RUN pip install python-multipart Pillow scipy

COPY ./app /app
