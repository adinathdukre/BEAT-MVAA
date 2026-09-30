FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=/app \
    MVAA_APP_DIR=/app \
    MVAA_WEIGHTS_DIR=/app/weights \
    MVAA_INPUT_DIR=/input \
    MVAA_OUTPUT_DIR=/output

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common
COPY task1 /app/task1
COPY task2 /app/task2
COPY task3 /app/task3
COPY configs /app/configs
COPY scripts /app/scripts
COPY weights /app/weights

ENTRYPOINT ["python", "/app/scripts/run_inference.py"]
