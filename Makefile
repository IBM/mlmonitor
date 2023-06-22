REPOSITORY=us.icr.io/mlops
NAME=mlmonitor-python
VERSION?=1.0.0
CONFIG_FILE?=/app/base/config.json

.PHONY: clean docker push run

docker:
	echo "${VERSION}" > ./version.meta
	rm -rf ./build  && rm -rf ./dist  && python setup.py bdist_wheel
	docker build --platform linux/amd64 --no-cache -t $(REPOSITORY)/$(NAME):$(VERSION) . -f Dockerfile --build-arg VERSION=${VERSION}
	rm -rf ./build  && rm -rf ./dist && rm -rf ./mlmonitor.egg-info

push:
	docker push $(REPOSITORY)/$(NAME):$(VERSION)

clean:
	docker rmi -f $(REPOSITORY)/$(NAME):$(VERSION)

run:
	docker run -it --platform linux/amd64 -v ${CONFIG_FILE}:/app/base/config.json:Z -e MONITOR_CONFIG_FILE=/app/base/config.json $(REPOSITORY)/$(NAME):$(VERSION)
