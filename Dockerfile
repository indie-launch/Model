FROM python:3.11-slim-bookworm

# --- OS deps ---
RUN apt-get update -y && apt-get upgrade -y && \
apt-get install -y --no-install-recommends \
curl ca-certificates bash git tini && \
rm -rf /var/lib/apt/lists/*


# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
cp /root/.local/bin/uv /usr/local/bin/uv


# Create an unprivileged user
RUN useradd -ms /bin/bash app


# Workdir
WORKDIR /app


# Caches for HF (optional but handy); override at runtime if desired
ENV HF_HOME=/app/.cache/huggingface \
TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
PIP_DISABLE_PIP_VERSION_CHECK=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1


# --- Dependency install with layer caching ---
# Prefer pyproject.toml + uv.lock if present; fallback to requirements.txt.
# Copy lock/manifest first so we can cache deps between code changes.
COPY pyproject.toml* uv.lock* requirements.txt* ./


# Create venv and install deps (handles either pyproject or requirements)
RUN uv venv /app/.venv && \
if [ -f uv.lock ] || [ -f pyproject.toml ]; then \
uv sync --frozen --python /app/.venv/bin/python; \
elif [ -s requirements.txt ]; then \
/app/.venv/bin/python -m pip install --upgrade pip && \
uv pip install -r requirements.txt --python /app/.venv/bin/python; \
else \
echo "No pyproject.toml/uv.lock or requirements.txt found" >&2; exit 1; \
fi


# --- App code ---
# Copy only what you need for runtime; dev images can copy the whole repo
COPY . /app


# Fix ownership and switch user
RUN chown -R app:app /app
USER app


# Use tini as PID 1 for clean signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
# Start a login shell with the venv activated by default
CMD ["/bin/bash", "-lc", "source /app/.venv/bin/activate && exec bash"]
