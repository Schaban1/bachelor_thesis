# Exploring Low-Dimensional Subspaces of Stable Diffusion Parameters

## Installation

We provide a `Dockerfile` that contains the required installation steps. If you want to install the required packages
manually via python, use the following command:

```
pip3 install -r requirements.txt
```

## Usage

A configuration file can be found under `configs/config.yaml`.

The web UI can be deployed using the following command:

```
python3 main.py
```

The web UI will be hosted under the port specified in the config.

It is recommended to run the backend using CUDA on a 20 GB GPU. Our deployment runs on a 20 GB MIG partition of an
NVIDIA A100 GPU.
