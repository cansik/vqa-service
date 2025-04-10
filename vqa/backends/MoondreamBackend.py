from typing import List

import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

from vqa.backends.BaseVQABackend import BaseVQABackend
from vqa.utils import torch_utils


class MoondreamBackend(BaseVQABackend):
    def __init__(self, model_name: str = "vikhyatk/moondream2", revision: str = "2025-01-09"):
        self.model_name = model_name
        self.device = torch_utils.get_device()
        self.dtype = torch_utils.get_dtype(allow_float_16=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
        image_embeds = self.model.encode_image(pil_image)

        results = []
        for i, question in enumerate(questions):
            answer = self.model.answer_question(image_embeds, question, self.tokenizer)
            results.append(answer.strip())

        return results

    @property
    def name(self) -> str:
        return f"Moondream with model {self.model_name}"