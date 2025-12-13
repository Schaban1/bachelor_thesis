FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/AI4LIFE-GROUP/SpLiCE.git && \
    cd SpLiCE && \
    pip install .

COPY . .

RUN pip3 install -r requirements.txt

RUN pip install datasets einops strenum wandb beartype
RUN pip install --no-deps sparse_autoencoder

ENTRYPOINT python3 main.py
