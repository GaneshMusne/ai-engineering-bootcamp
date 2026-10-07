FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app

# Copy workspace root files for dependency resolution
COPY pyproject.toml uv.lock ./

# Copy workspace manifests before source so code changes reuse dependencies
COPY apps/api/pyproject.toml ./apps/api/pyproject.toml
COPY apps/chatbot_ui/pyproject.toml ./apps/chatbot_ui/pyproject.toml

ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

# Install dependencies including workspace packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api --no-install-workspace

COPY apps/api ./apps/api

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api

# Enable bytecode compilation and Python optimization
ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/tmp/huggingface

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/api/src"

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose the FastAPI port
EXPOSE 8000

WORKDIR /app/apps/api/src

# Command to run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
