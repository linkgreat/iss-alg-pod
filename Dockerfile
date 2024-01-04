FROM f7star/python-cv AS prepare
LABEL authors="fjw"

WORKDIR /scripts
COPY . .
RUN pip install -r requrirements.txt


ENTRYPOINT ["top", "-b"]