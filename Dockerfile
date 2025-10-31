# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

# native utils commonly needed (safe + small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# use your pinned container requirements file
COPY requirements_container.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy the code
COPY . /app

# expose port (optional)
EXPOSE 8000

# Start the FastAPI app; change this line if your entrypoint differs
# e.g., if your app is in web/main.py use: web.main:app
CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST} --port ${APP_PORT}"]
