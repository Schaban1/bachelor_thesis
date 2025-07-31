FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT python3 main.py
