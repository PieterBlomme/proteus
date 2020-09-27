FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt ./requirements.txt
#Install nvidia-pyindex first to have access to nvidia pypi
RUN pip install nvidia-pyindex==1.0.4
RUN pip install -r requirements.txt

COPY ./app /app
