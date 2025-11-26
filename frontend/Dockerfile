FROM python:3.13-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements-frontend.txt

EXPOSE 8050

CMD ["python", "frontend/frontend.py"]
