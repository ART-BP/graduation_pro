# syntax=docker/dockerfile:1

ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${ISAAC_LAB_IMAGE} AS base

ARG GO2W_IMAGE_VERSION=0.1.0
ARG GIT_COMMIT=unknown

LABEL org.opencontainers.image.title="Go2W Terrain Planner" \
      org.opencontainers.image.version="${GO2W_IMAGE_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      com.nvidia.isaac.lab.version="2.3.2" \
      com.nvidia.isaac.sim.version="5.1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GO2W_PROJECT_ROOT=/workspace/go2w_terrain_planner \
    GO2W_DATA_ROOT=/workspace/data \
    GO2W_IMAGE_VERSION=${GO2W_IMAGE_VERSION} \
    GO2W_GIT_COMMIT=${GIT_COMMIT} \
    ISAAC_LAB_VERSION=2.3.2 \
    ISAAC_SIM_VERSION=5.1.0

WORKDIR ${GO2W_PROJECT_ROOT}

COPY source ./source
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs
COPY pyproject.toml README.md ./

RUN /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
      -e source/go2w_terrain_planner && \
    /workspace/isaaclab/isaaclab.sh -p -m pip install --no-cache-dir \
      pytest==8.3.5 onnx==1.20.1 onnxruntime==1.20.1 && \
    mkdir -p /workspace/data/runs /workspace/data/export

COPY docker/entrypoint.sh /usr/local/bin/go2w-entrypoint
RUN chmod +x /usr/local/bin/go2w-entrypoint

FROM base AS unit-test
ENV GO2W_SKIP_TASK_REGISTRATION=1
RUN /workspace/isaaclab/isaaclab.sh -p -m pytest -q tests/unit

FROM base AS runtime
ENTRYPOINT ["/usr/local/bin/go2w-entrypoint"]
CMD ["shell"]
