FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8000
CMD ["uvicorn", "safesite_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
