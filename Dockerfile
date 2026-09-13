FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline pipeline
COPY sql sql
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "pipeline", "status"]
