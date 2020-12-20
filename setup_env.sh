docker stop "$(docker ps | grep pycaret | awk '{print $NF}' | tail -n 1)"
docker rmi -f pycaret
docker build . -t pycaret
docker run -v "$PWD":/app -dit -p 8888:8888 pycaret
docker exec "$(docker ps | grep pycaret | awk '{print $NF}' | tail -n 1)" \
python -m jupyterlab --port=8888 --no-browser --ip=0.0.0.0 --allow-root