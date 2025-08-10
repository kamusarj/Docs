FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /dl

COPY train_fasterrcnn.py train.py
COPY voc_dataset.py voc_dataset.py

RUN apt-get update
RUN apt-get install vim -y

RUN pip install tensorboard torchmetrics

