# Docker deployment


## Issues

Azure cache docker images. If given node is in operation (there was no downtime) cached image will be deployed.\ 
Even after remove and upload new docker image to `Azure registry` cached image can be deployed.\ 
To avoid this situation always increase image `tag`.\
__Do not use `latest` tag__


### Naming convention

Production images should be labeled as `image:tag`: computation_worker:<version>-prod  e.g: `computation_worker:1.0.5-prod`\
Development images should be labeled as `image:tag`: computation_worker<version>-dev   e.g: `computation_worker:1.0.6-dev` 

## Job Worker

Worker is responsible for performing certain job scheduled from `virtum_clb_server`.\
Each job inside container is done by one of `virtules` scripts.

### Create and upload to registry
Login to docker registry: `docker login mitcldx.azurecr.io -p <pass> -u <user>`\
Build docker image: `docker build -t mitcldx.azurecr.io/<image>:<tag> . -f docker/virtum/Dockerfile --no-cache`\
Upload to registry: `docker push mitcldx.azurecr.io/<image>:<tag>`