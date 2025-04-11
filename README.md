# Visual Question Answering (VQA) Service

## Overview

The VQA Service is a machine learning based application that allows users to ask questions about images and receive answers.
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

## Supported Models

The following VLM backends are supported:

| Backend ID | Model | Description |
|------------|-------|-------------|
| `blip` | Salesforce/blip-vqa-base | BLIP base model for visual question answering |
| `blip2` | Salesforce/blip2-opt-2.7b | BLIP2 with OPT 2.7B language model |
| `blip2-flan` | Salesforce/blip2-flan-t5-xl | BLIP2 with Flan-T5-XL language model |
| `vilt` | dandelin/vilt-b32-finetuned-vqa | ViLT model fine-tuned for VQA tasks |
| `vlmmlx` | mlx-community/Qwen2-VL-2B-Instruct-4bit | Default MLX-based VLM for Apple Silicon |
| `vlmmlx-phi35` | mlx-community/Phi-3.5-vision-instruct-4bit | Phi-3.5 Vision model optimized for MLX |
| `vlmmlx-smolvlm2` | mlx-community/SmolVLM2-500M-Video-Instruct-mlx-8bit-skip-vision | SmolVLM2 optimized for MLX |
| `namo` | - | Namo VLM model |
| `moondream` | vikhyatk/moondream2 | Moondream2 model with GPU support |
| `moondream-cpu` | vikhyatk/moondream2 | Moondream2 model optimized for CPU inference |
| `smolvlm` | HuggingFaceTB/SmolVLM-256M-Instruct | Lightweight VLM model |
| `smolvlm2` | HuggingFaceTB/SmolVLM2-256M-Video-Instruct | SmolVLM2 with video instruction capabilities |


Each backend offers different capabilities and performance characteristics.
