# Демон мозга мухи. Тонкий образ: в рантайме нужен только numpy.
FROM python:3.11-slim

WORKDIR /app

# pandas тут не нужен: коннектом собирается один раз снаружи в .npz,
# а рантайм читает готовый numpy-архив
RUN pip install --no-cache-dir "numpy>=1.21"

COPY flybrain/ /app/flybrain/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

ENV FLYBRAIN_HOST=0.0.0.0 \
    FLYBRAIN_PORT=5665 \
    PYTHONUNBUFFERED=1

EXPOSE 5665

# data/ монтируется томом: коннектом весит сотни мегабайт,
# в образ его класть незачем
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:5665/status',timeout=4)"

CMD ["python3", "-m", "flybrain.server"]
