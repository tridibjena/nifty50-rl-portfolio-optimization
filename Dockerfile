# Reproducible environment for the full pipeline.
#
#   docker build -t nifty-rl .
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/assets:/app/assets" nifty-rl
#
# The data snapshot is pinned via DataConfig.end_date, so a container run in six months
# reproduces the published figures rather than silently evaluating a different period.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src

WORKDIR /app

# Core scientific stack first so it caches independently of the source tree.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install \
      numpy pandas scipy scikit-learn pyarrow yfinance matplotlib seaborn pytest

COPY pyproject.toml conftest.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Fail the build if the suite fails -- an image that ships broken code is worse than
# no image.
RUN python -m pytest tests/ -q

CMD ["python", "scripts/run_pipeline.py"]
