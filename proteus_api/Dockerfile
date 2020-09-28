FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY proteus_api/requirements.txt ./requirements.txt
#Install nvidia-pyindex first to have access to nvidia pypi
RUN pip install nvidia-pyindex==1.0.4
RUN pip install -r requirements.txt

#Install model packages
COPY ./packages /packages
RUN pip install -r /packages/package_install.txt

COPY proteus_api/app /app
