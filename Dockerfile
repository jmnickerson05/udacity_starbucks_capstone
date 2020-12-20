FROM python:3.7-slim

RUN pip install pycaret
RUN pip install jupyterlab
RUN apt-get update -y && apt-get upgrade -y && apt-get install libgomp1
WORKDIR /app
ADD . /app

#CMD ["python", "-m", "jupyterlab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
#CMD ["jupyter", "notebook", "list"]
