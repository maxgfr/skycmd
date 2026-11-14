# s6-secrets v2 (docs/s6-secrets2-prereg.md, section 6): the environment of
# cbx-secrets-detector-17M. The model's own code (cbx_secrets.py, downloaded from its
# repository) only ever runs here: no network at run time, the model folder and the folders
# of the split mounted read-only, one output folder. Python packages are those of
# scripts/secrets2/envs/cbx/uv.lock, CPU build of torch.
ARG PYTHON
ARG UV
FROM ${UV} AS uv
FROM ${PYTHON}
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
WORKDIR /env
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev && rm /usr/local/bin/uv
ENV PATH=/opt/venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
USER 65534:65534
WORKDIR /work
