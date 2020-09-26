FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install nvidia-pyindex
RUN pip install tritonclient[http,grpc]

COPY ./app /app
