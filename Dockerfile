FROM f7star/python-cv:3.18.8-slim AS prepare
LABEL authors="fjw"

WORKDIR /scripts
COPY . .
RUN pip install -r requirements.txt

