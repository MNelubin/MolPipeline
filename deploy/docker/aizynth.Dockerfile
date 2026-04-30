FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-aizynth.txt /app/requirements-aizynth.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-aizynth.txt

COPY mvp /app/mvp

EXPOSE 8052

CMD ["python", "-m", "uvicorn", "mvp.aizynth_service:app", "--host", "0.0.0.0", "--port", "8052"]
