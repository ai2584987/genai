FROM python:3.12-slim

# uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1

WORKDIR /code

# 1) Dependencies first (cached unless pyproject/uv.lock change)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 2) Then the code and the trained weights
COPY app ./app
COPY helper_lib ./helper_lib

EXPOSE 80
CMD ["/code/.venv/bin/fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]
