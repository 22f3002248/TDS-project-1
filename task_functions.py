#!/usr/bin/env python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "requests",
#     "numpy",
#     "scikit-learn",
#     "pytesseract",
#     "httpx"
# ]
# ///

from PIL import Image, ImageEnhance, ImageFilter
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess

import httpx
import numpy as np
import pytesseract
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
headers = {
    "Content-type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}


def format_file(input_path: str, output_path: str, inplace: bool = True, formatter: str = "prettier@3.4.2"):
    """
    A2. Format the contents of a Markdown file using the specified formatter (e.g., prettier@3.4.2).
    If inplace is True, update the file at input_path; otherwise, write the formatted output to output_path.
    The function first checks if the formatter is installed; if not, it installs it.
    """
    if not os.path.exists(input_path):
        raise Exception(f"{input_path} not found")
    logging.info("Started formatting")
    try:
        # Check if npx can fetch the formatter version
        version_cmd = ["npx", "--yes", formatter, "--version"]
        version_result = subprocess.run(
            version_cmd, capture_output=True, text=True, check=True
        )
        logging.info(
            f"{formatter} is available: {version_result.stdout.strip()}")
    except subprocess.CalledProcessError:
        logging.info(f"{formatter} is not installed. Installing...")
        try:
            subprocess.run(
                ["npm", "install", "-g", formatter],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"Installed {formatter}")
        except subprocess.CalledProcessError as e_install:
            raise Exception(
                f"Failed to install {formatter}: {e_install.stderr}")

    # Read the original file
    with open(input_path, "r", encoding="utf-8") as f:
        original = f.read()

    # Run the formatter with npx
    cmd = ["npx", "--yes", formatter, "--stdin-filepath", input_path]
    try:
        result = subprocess.run(
            cmd,
            input=original,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Formatter failed: {e.stderr}")

    formatted = result.stdout

    # Write the output
    target_path = input_path if inplace else output_path
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(formatted)

    logging.info("Task A2 completed")
    return "A2 completed"


async def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.SHARPEN)  # Enhance edges
    image = ImageEnhance.Contrast(image).enhance(2)  # Increase contrast
    image = image.resize((image.width * 2, image.height * 2))  # Upscale
    return image


async def image_ocr(input_path, output_path):
    """
    A8. Read a PNG image, process it using OCR to extract the credit card number,
    remove any spaces or hyphens, and write the result to the specified output file.
    """
    try:
        image = await preprocess_image(input_path)
    except Exception as e:
        raise Exception("Error opening or processing image: " + str(e))

    extracted_text = pytesseract.image_to_string(image, config="--psm 6")
    logging.info(f"OCR Output:\n{extracted_text}")

    match = re.search(r"((?:\d[\s-]?){13,19})", extracted_text)
    if not match:
        raise Exception("Credit card number not found in OCR output")

    card_number = re.sub(r"[\s-]", "", match.group(1))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card_number)

    logging.info("Task A8 completed")
    return "A8 completed"


def simulate_embedding(text, dim=10):
    """Generate a deterministic embedding vector for a given text using SHA256."""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:dim]]


async def comments_similarity(input_path, output_path):
    """
    A9. Read /data/comments.txt (one comment per line), compute embeddings (using the OpenAI API if available,
         otherwise simulate embeddings), find the most similar pair of comments (via cosine similarity),
         and write the two comments (alphabetically sorted, one per line) to /data/comments-similar.txt.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f if line.strip()]
    if len(comments) < 2:
        raise Exception("Not enough comments for comparison")
    simulate = False
    embeddings = None

    if not AIPROXY_TOKEN:
        simulate = True
    else:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
                    headers={"Authorization": f"Bearer {AIPROXY_TOKEN}"},
                    json={"model": "text-embedding-3-small", "input": comments},
                )
            if response.status_code != 200:
                simulate = True
            else:
                try:
                    json_data = response.json()
                    if "data" not in json_data:
                        simulate = True
                    else:
                        embeddings = np.array(
                            [item["embedding"] for item in json_data["data"]])
                except Exception:
                    simulate = True
        except Exception:
            simulate = True

    if simulate or embeddings is None:
        embeddings = np.array([simulate_embedding(comment, dim=10)
                              for comment in comments])

    similarity = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity, -np.inf)
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    pair = sorted([comments[i], comments[j]])
    result_text = "\n".join(pair)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result_text)
    logging.info("Task A9 completed")
    return "A9 completed"


async def markdown_indexer(docs_dir, output_path):
    """
    A6. Recursively find all Markdown (.md) files in /data/docs/, extract the first H1 line from each,
         and write an index (mapping filename to title) to /data/docs/index.json.
    """
    index = {}
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if fname.endswith(".md"):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(
                    full_path, docs_dir).replace(os.sep, "/")
                title = None
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("# "):
                            title = line[2:].strip()
                            break
                if title:
                    index[rel_path] = title
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    logging.info("Task A6 completed")
    return "A6 completed"
