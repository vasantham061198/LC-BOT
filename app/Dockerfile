FROM python:3.10

WORKDIR /python-app

COPY . .

RUN pip3 install -r requirements.txt

CMD [ "waitress-serve", "--port", "8888", "--call", "server.app:generate_app"]
