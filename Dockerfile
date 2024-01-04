FROM f7star/python:3.18.8-slim AS prepare
LABEL authors="fjw"

WORKDIR /scripts
COPY . .
RUN pip install -r requrirements.txt


ENTRYPOINT ["top", "-b"]