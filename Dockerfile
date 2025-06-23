FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git
COPY . .
RUN mkdir -p data models index_store
EXPOSE 8000
# CMD ["python", "app.py"]
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "app:app"]
