FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr make && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir .[app,tables,ocr]
EXPOSE 8000
CMD ["uvicorn", "mmrag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
