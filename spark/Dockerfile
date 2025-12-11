FROM spark:4.0.1-python3

WORKDIR /app
ENV HOME=/app

RUN mkdir -p "$HOME/.local/bin" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="$HOME/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYSPARK_PYTHON="$VIRTUAL_ENV/bin/python"
ENV PYSPARK_DRIVER_PYTHON="$VIRTUAL_ENV/bin/python"

# --- NEW: Spark config for Ivy cache ---
RUN mkdir -p /app/conf \
    && printf 'spark.jars.ivy /app/.ivy2\n' > /app/conf/spark-defaults.conf

ENV SPARK_CONF_DIR=/app/conf
