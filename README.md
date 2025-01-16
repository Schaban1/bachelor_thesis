# Multimodal Machine Learning Lab WiSe24/25 - Project

This repository contains code and material for the project in the Multimodal Machine Learning Lab course in winter term 24/25 at the University of Kassel.

## Structure

- `docs`: Contains everything documentation-related, like sketches from the planning phase and links relevant for developers.
- `prototype`: Contains the code for the (currently WIP) prototype of a probability-based image generation system. Further information can be viewed in the `prototype`-README.

## Installation

**Python Virtual Environment**

Create venv: ` python -m venv MMML_venv` (is ignored by git)

Activate venv: `source MMML_venv/bin/activate`

Install requirements: `pip install -r requirements.txt`

## General Links

- [Course Page](https://temir.org/teaching/multimodal-machine-learning-ws24/multimodal-machine-learning-ws24.html)


# Install dependencies
I ran into some issues when installing the dependencies. Here are some notes on how to solve them (on macOS).
## xformers 
cf. https://stackoverflow.com/questions/60005176/how-to-deal-with-clang-error-unsupported-option-fopenmp-on-travis (16.01.25)
1. run ```pip install xformers --no-dependencies --no-cache-dir```
produces error: 
`clang: error: unsupported option '-fopenmp'
      error: command '/usr/bin/gcc' failed with exit code 1
`

2a. install llvm via homebrew: ```brew install llvm libomp```
read console output and add the following to your shell profile (e.g. .zshrc):
```shell
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```
(possibly smart, but I did not and obtained error: 
`python setup.py bdist_wheel did not run successfully.`

2b. Afterwards, I ran the following command:
```shell
export CC=/opt/homebrew/opt/llvm/bin/clang           
export CXX=/opt/homebrew/opt/llvm/bin/clang++
```
and then the following command:
```shell
pip install xformers --no-dependencies --no-cache-dir
```
This worked for me.
)