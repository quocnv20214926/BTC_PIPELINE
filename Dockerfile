FROM python:3.12-slim
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY pipeline pipeline
COPY sql sql
COPY model_training/modelv2.py model_training/modelv2.py
COPY model_training/strategy_h1.py model_training/strategy_h1.py
COPY model_training/positioning_data.py model_training/positioning_data.py
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "pipeline", "status"]
