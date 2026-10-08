FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /code
COPY dashboard/requirements.txt .
RUN pip install -r requirements.txt
COPY dashboard ./dashboard
RUN useradd --create-home appuser && chown -R appuser /code
USER appuser
EXPOSE 8501
CMD ["sh", "-c", "streamlit run dashboard/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
