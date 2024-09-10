#!/bin/bash
case "$1" in
    BASE)
	sudo docker buildx build -t mlm-pipe-ubuntu24.04-cuda12.04:base -f Dockerfile_base .
        ;;

    MLM)
	sudo docker buildx build --no-cache -t mlm-pipe-ubuntu24.04-cuda12.04:mlm -f Dockerfile_mlm .
        ;;  
    *)
	break
        ;;  
esac
sudo docker buildx build --no-cache -t mlm-pipe-ubuntu24.04-cuda12.04:client -f Dockerfile .
