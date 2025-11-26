FROM python:3.13-bullseye

WORKDIR /app

# Install OS deps for WeasyPrint
RUN apt-get update \
    && apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libjpeg-dev fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements-backend.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

