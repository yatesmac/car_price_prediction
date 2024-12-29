FROM python:3.12-slim

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY data ./data

COPY models ./models

COPY src ./src

EXPOSE 9696

WORKDIR /app/src/app

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]