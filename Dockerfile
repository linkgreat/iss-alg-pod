FROM f7star/python:3.18.8-slim AS prepare
LABEL authors="fjw"

WORKDIR /scripts
COPY . .
RUN pip install -r requirements.txt


ENTRYPOINT ["top", "-b"]