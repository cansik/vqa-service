from typing import List

import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from vqa.utils import torch_utils


class Blip2Backend:
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        self.device = torch_utils.get_device_string()

        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        results = []
        for i, q in enumerate(questions):
            inputs = self.processor(images=image, text=f"Question: {q} Answer:", return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**inputs)
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            results.append(result)

        return results
