#!/bin/bash
export LD_LIBRARY_PATH=/home/ubuntu/embedding/venv/lib/python3.12/site-packages/nvidia/cufft/lib:/home/ubuntu/embedding/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "Starting Uvicorn with TensorRT support..."
/home/ubuntu/embedding/venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 5000
