# syntax=docker/dockerfile:1
FROM python:3.12.16-slim-trixie

WORKDIR /app

# Dependencias primero (capa cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

# Usuario no-root
RUN useradd --uid 10001 --no-create-home appuser
USER 10001

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
