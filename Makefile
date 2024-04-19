.PHONY: stop, start, remove, open, build

SERVICE_LIST = router yolo llava
GPU_OPTIONS=--gpus all

# env variables for use in typefly
export IS_LOCAL_YOLO_SERVICE=false

validate_service:
ifeq ($(filter $(SERVICE),$(SERVICE_LIST)),)
	@echo Invalid SERVICE: [$(SERVICE)], valid values are [$(SERVICE_LIST)]
	$(error Invalid SERVICE, valid values are [$(SERVICE_LIST)])
endif

stop: validate_service
	@echo "=> Stopping typefly-$(SERVICE)..."
	@-docker stop -t 0 typefly-$(SERVICE) > /dev/null 2>&1
	@-docker rm -f typefly-$(SERVICE) > /dev/null 2>&1

start: validate_service
	@make stop SERVICE=$(SERVICE)
	@echo "=> Starting typefly-$(SERVICE)..."
	docker run -td --privileged --ipc=host $(GPU_OPTIONS) --net=host \
		--env-file ./docker/env.list \
    	--name="typefly-$(SERVICE)" typefly-$(SERVICE):0.1
# -p 50050:50050 \
# -p 50049:50049
# -p 8888:8888
# --net=host

remove: validate_service
	@echo "=> Removing typefly-$(SERVICE)..."
	@-docker image rm -f typefly-$(SERVICE):0.1  > /dev/null 2>&1
	@-docker rm -f typefly-$(SERVICE) > /dev/null 2>&1

open: validate_service
	@echo "=> Opening bash in typefly-$(SERVICE)..."
	@docker exec -it typefly-$(SERVICE) bash

build: validate_service
	@echo "=> Building typefly-$(SERVICE)..."
	@make stop SERVICE=$(SERVICE)
	@make remove SERVICE=$(SERVICE)
	@echo -n "=>"
	docker build -t typefly-$(SERVICE):0.1 -f ./docker/$(SERVICE)/Dockerfile .
	@echo -n "=>"
	@make start SERVICE=$(SERVICE)

typefly:
	bash ./serving/webui/install_requirements.sh
	cd ./proto && bash generate.sh
	python3 ./serving/webui/typefly.py --use_http --llava_prefix --llava_bounding_boxes
# Available feature flags:
# --use_virtual_cam : Use virtual camera instead of real camera
# --use_http: Use http instead of grpc (should be enabled more or less always)
# TODO in progress:
# --llava_prefix : attaches a prefix to requests sent to llava to optimize the output.
# --llava_bounding_boxes : sends yolo bounding-boxes and labels to llava
# --replan-skill : enables the "re-plan" skill of TypeFly
