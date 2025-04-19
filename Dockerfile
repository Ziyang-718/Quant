FROM tensorflow/tensorflow:2.5.0-gpu
WORKDIR /app
COPY . .
RUN pip install bert4keras -t bert4keras/
RUN pip install -r requirements.txt
