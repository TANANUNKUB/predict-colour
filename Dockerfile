FROM python:3.10
WORKDIR /code 
COPY . . 
RUN chmod 777 ./public
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 80
CMD ["python", "main.py"]
