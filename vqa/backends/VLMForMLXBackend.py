# pip install mlx-vlm
from typing import List

import numpy as np
from PIL import Image

from vqa.backends.BaseVQABackend import BaseVQABackend


class VLMForMLXBackend(BaseVQABackend):

    def __init__(self, model_name: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model_name = model_name

        self.model, self.processor = load(model_name, trust_remote_code=True)
        self.config = load_config(model_name, trust_remote_code=True)

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        from mlx_vlm import generate, apply_chat_template

        pil_image = Image.fromarray(np.uint8(image)).convert("RGB")

        results = []
        for i, question in enumerate(questions):
            formatted_prompt = apply_chat_template(
                self.processor, self.config, question, num_images=1
            )

            output = generate(self.model, self.processor, formatted_prompt, pil_image, verbose=False)
            if isinstance(output, tuple):
                output = output[0]

            results.append(output.strip())

        return results

    @property
    def name(self) -> str:
        return f"VLM MLX with model {self.model_name}"
