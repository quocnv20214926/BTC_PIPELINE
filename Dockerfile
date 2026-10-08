FROM python:3.12-slim
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY pipeline pipeline
COPY sql sql
COPY model_training/modelv2.py model_training/modelv2.py
COPY model_training/execution_m1.py model_training/execution_m1.py
COPY model_training/modelv3.py model_training/modelv3.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "pipeline", "status"]
