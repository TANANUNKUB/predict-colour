docker stop $(docker ps -a -q)

docker rm $(docker ps -a -q) -f

docker rmi $(docker images -q) -f

docker build -t predict-colour .

docker tag predict-colour:latest predict-colour:v1.0.0

docker run -d -p 5000:5000 predict-colour:v1.0.0

echo "Kungfu server is running on http://localhost:5000"

$SHELL