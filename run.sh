docker pull python:3.10

docker build --tag predict-colour .

docker tag predict-colour:latest predict-colour:v1.0.0

docker run --name predict-colour-docker -p 80:80 -d predict-colour:v1.0.0

echo "server is running on http://localhost:80"
