FROM python:3.10
WORKDIR /code 
COPY . . 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["bash"]
