FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-min.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Copy code + config
COPY src /app/src
COPY datasets /app/datasets
COPY data /app/data
COPY scripts /app/scripts

ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "mch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
