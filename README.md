# Distributed Systems Lab
``
docker build -t ml-deploy:latest .

docker run -p 3000:3000 -v ../uploads:/app/uploads ml-deploy:latest

``
### details are in '/home'