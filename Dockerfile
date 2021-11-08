FROM python:3.8-slim-buster AS builder

RUN apt-get update && apt-get -y install git

WORKDIR /app

COPY . /app

# Update PIP & install package/requirements
RUN python -m pip install --upgrade pip
RUN pip install -e .
RUN pip install --upgrade tensorflow-hub

# Execute the machine learning pipeline:
CMD python pipeline.py
