FROM python:3.11
ARG VERSION
ENV WHL_FILE=mlmonitor-${VERSION}-py3-none-any.whl
ADD ./dist/${WHL_FILE} /tmp/${WHL_FILE}
ADD ./mlmonitor/credentials_example.cfg /app/base/config.json

RUN apt-get clean && apt-get -y update && \
    apt-get install -yq less vim jq zip && \
    pip install --upgrade pip setuptools wheel && \
    pip install "docutils>=0.14,<0.16" python-dotenv && \
    pip install "/tmp/"${WHL_FILE}"[local]"

ENTRYPOINT ["/bin/bash"]
