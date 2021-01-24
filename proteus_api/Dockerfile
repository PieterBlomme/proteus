FROM python:3.7

RUN apt-get clean

RUN pip install fastapi uvicorn

COPY proteus_api/requirements.txt ./requirements.txt

# to be sure, old pip resolver
RUN python -m pip install --upgrade pip==20.2

#Install nvidia-pyindex first to have access to nvidia pypi
RUN pip install nvidia-pyindex==1.0.4
RUN pip install -r requirements.txt

EXPOSE 80

COPY proteus_api/app /app

CMD ["sh","-c","pip install -r /packages/package_install.txt && uvicorn app.main:app --reload --host 0.0.0.0 --port 80"]