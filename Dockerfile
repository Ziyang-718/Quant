FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
