# syntax=docker/dockerfile:1

ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2

FROM ${ISAAC_LAB_IMAGE} AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace/go2w_terrain_planner

COPY pyproject.toml ./
COPY src ./src

RUN /workspace/isaaclab/isaaclab.sh \
    -p -m pip install \
    --no-cache-dir \
    -e .

RUN /workspace/isaaclab/isaaclab.sh \
    -p -m pip install \
    --no-cache-dir \
    pytest

COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
COPY docker/entrypoint.sh /usr/local/bin/go2w-entrypoint

RUN chmod +x /usr/local/bin/go2w-entrypoint


FROM base AS unit-test

RUN /workspace/isaaclab/isaaclab.sh \
    -p -m pytest \
    -q /workspace/go2w_terrain_planner/tests/unit


FROM base AS runtime

RUN cd /workspace/isaaclab && \
    ./isaaclab.sh -i rsl_rl

ENTRYPOINT ["/usr/local/bin/go2w-entrypoint"]
CMD ["shell"]