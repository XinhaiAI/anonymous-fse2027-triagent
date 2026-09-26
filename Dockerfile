# ==============================================================================
# ACM FSE 2027 Replication Package Dockerfile
# Paper: Guarding the Regulated Frontier: Dual-Track Tri-Agent Semantic Entropy
#        for Hallucination Mitigation in High-Stakes Medical Device Compliance
# ==============================================================================

# Official ultra-lightweight Python 3.11 base image
FROM python:3.11-slim

# Set environment variables for clean python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# Create application directory
WORKDIR /app

# Install minimal runtime dependencies (pure standard library fallback supported)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy replication package artifacts
COPY medcred_bench_lite.json .
COPY calibration_bins.json .
COPY telemetry_latency_rtx4090.json .
COPY profile_latency.py .
COPY run_demo.py .
COPY README.md .

# Create non-root user for security and container compliance
RUN useradd -m -u 1000 replicator && \
    chown -R replicator:replicator /app
USER replicator

# Default container entrypoint executing benchmark demo
CMD ["python", "run_demo.py", "--benchmark", "medcred_bench_lite.json"]
