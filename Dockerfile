FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY copilot ./copilot
COPY scripts ./scripts

# rules.json и taxonomy.json монтируются в /data вместе с обращениями
ENV COPILOT_DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
