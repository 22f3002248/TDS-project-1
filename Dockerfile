FROM python:3.12-slim-bookworm

# Install dependencies in one layer and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      tesseract-ocr \
      nodejs \
      npm && rm -rf /var/lib/apt/lists/*
RUN npm install -g npx
# Download the latest installer for uv and run it, then remove the installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Create the /data directory
RUN mkdir -p /data

# Copy your application code
COPY app.py /
COPY task_functions.py /
# Run the app with uv
CMD ["uv", "run", "app.py"]
