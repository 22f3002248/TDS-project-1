FROM python:3.12-slim-bookworm

# Install dependencies in one layer and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      tesseract-ocr \
      nodejs \
      npm && rm -rf /var/lib/apt/lists/*

# Install npx globally
RUN npm install -g npx

# Download and install uv, then clean up the installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Create necessary directories
RUN mkdir -p /app /data

# Set the working directory
WORKDIR /app


# --- 🛠️ Prepare node_modules before runtime ---
# Create an empty package.json to initialize node_modules
RUN npm init -y

# Set correct ownership and permissions
RUN chown -R root:root /app/node_modules && chmod -R 777 /app/node_modules


# Copy application files
COPY app.py .
COPY task_functions.py .

# Ensure `tesseract-ocr` is updated **at build time** (not every time the container runs)
RUN apt-get update && apt-get upgrade -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Start the app
CMD ["uv", "run", "app.py"]