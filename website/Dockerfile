FROM python:3.11-slim

# opencv-python-headless still dynamically links a couple of system libs on
# a slim base even though it's the "headless" build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY website/requirements.txt ./website-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r website-requirements.txt

COPY solution.py ./
COPY src/ ./src/
COPY website/ ./website/
COPY weights/ ./weights/

# demo_server.py reads PORT from the environment (default 5000) --
# Render/Railway/HF Spaces/etc. all inject their own PORT at runtime, so
# nothing platform-specific is hardcoded here. EXPOSE is documentation only.
EXPOSE 5000

CMD ["python", "website/demo_server.py"]
