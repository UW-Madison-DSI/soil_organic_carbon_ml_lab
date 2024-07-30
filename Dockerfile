

FROM python:3.8-slim-buster

WORKDIR /app

COPY api/requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY api/ ./api

COPY api/model.pkl ./api/model.pkl
EXPOSE 8001
ENV RS_CONNECT_API_KEY="XXXXXXXX"

CMD ["rsconnect-python", "deploy", "fastapi", "api/main.py", "--server", "https://connect.doit.wisc.edu/", "--api-key", "$RS_CONNECT_API_KEY"]
