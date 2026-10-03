# GPU image for benchmarks, training and serving.
# The base image ships PyTorch built against the bundled CUDA runtime; the host
# only needs an NVIDIA driver and the NVIDIA container toolkit.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# torch.compile generates and compiles C/C++ launchers at runtime, so the
# runtime image needs a host compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Install dependencies before copying the source so code edits don't bust the layer cache.
COPY pyproject.toml README.md ./
COPY src/qnnbench/__init__.py src/qnnbench/__init__.py
RUN pip install --no-cache-dir -e ".[bench,serve]"
COPY src ./src

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data /app/results \
    && chown -R app /app
USER app

ENV PYTHONUNBUFFERED=1
# Default: full benchmark run. Override the command for training or serving, e.g.
#   docker run --gpus all -p 8000:8000 IMAGE uvicorn qnnbench.serve:app --host 0.0.0.0
CMD ["python", "-m", "qnnbench.bench", "all", "--out", "/app/results"]
