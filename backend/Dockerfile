# API image. TODO(SCRUM-38, Chi): add the [ml] extra (CPU torch) once the detector is wired in.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml config.py ./
COPY ml ml
COPY app app
COPY sim sim
COPY edge edge
COPY run run
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["python", "-m", "run.serve"]
