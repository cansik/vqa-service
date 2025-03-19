# Visual Question Answering (VQA) Service

## Overview

The VQA Service is a machine learning application that allows users to ask questions about images and receive answers.
It leverages state-of-the-art models to process images and generate accurate responses to user queries. The service is
designed to be flexible, supporting multiple backend models for different use cases.

## Installation

To install and set up the VQA Service, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cansik/vqa-service.git
   cd vqa-service
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the VQA Service, use the following command:

```bash
python -m vqa
```

### Command-line Arguments

- `--host`: Specify the service host (default: `127.0.0.1`).
- `--port`: Specify the service port (default: `7840`).
- `--backend`: Choose the VQA backend model (default: `blip`).

### Example

To run the service on a specific host and port with a chosen backend, use:

```bash
python -m vqa --host 0.0.0.0 --port 8000 --backend blip2
```

### Accessing the Service

Once the service is running, you can access it through a web interface provided by Gradio. Open your web browser and
navigate to `http://<host>:<port>` to interact with the service.

## Backends

The service supports multiple backend models, including:

- `blip`
- `blip2`
- `blip2-flan`
- `vilt`
- `moondream`

Each backend offers different capabilities and performance characteristics.
