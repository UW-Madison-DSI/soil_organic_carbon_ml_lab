

FROM python:3.8-slim-buster

WORKDIR /app

COPY api/requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY api/ ./api

COPY api/model.pkl ./api/model.pkl
# PORT
EXPOSE 8001
# Agrega las credenciales de RStudio Connect (reemplaza con tus propias credenciales)
ENV RS_CONNECT_API_KEY="XXXXXXXX"

# Configura el comando de despliegue como entrada principal
CMD ["rsconnect-python", "deploy", "fastapi", "api/main.py", "--server", "https://connect.doit.wisc.edu/", "--api-key", "$RS_CONNECT_API_KEY"]
