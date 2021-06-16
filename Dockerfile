FROM python:3.8-slim-buster

WORKDIR /app

COPY environment.yml .

COPY model ./model

COPY /api/app.py .

EXPOSE 5000

RUN pip install numpy \
                pandas \
                matplotlib \
                flask \
                tensorflow==2.5 \
                flask_restplus \
                Werkzeug==0.16.1

ENTRYPOINT ["flask","run","--host=0.0.0.0"]

