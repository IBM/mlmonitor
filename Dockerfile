FROM python:3.10
ARG VERSION
ENV WHL_FILE=mlmonitor-${VERSION}-py3-none-any.whl
ADD ./dist/${WHL_FILE} /tmp/${WHL_FILE}
ADD ./mlmonitor/credentials_example.cfg /app/base/config.json

RUN apt-get clean && apt-get -y update && \
    apt-get install -yq less && \
    apt-get install -yq vim && \
    apt-get install -yq jq && \
    apt-get install -yq zip && \
    pip install --upgrade pip && \
    pip install "/tmp/"${WHL_FILE}"[local,sagemaker,drift]"

ENTRYPOINT ["/bin/bash"]
