from typing import List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from vqa.backends.BaseVQABackend import BaseVQABackend


class MoondreamCpuBackend(BaseVQABackend):
    def __init__(self, model_name: str = "moondream-0_5b-int8.mf.gz", repo_id: str = "vikhyatk/moondream2"):
        import moondream as md
        self.model_name = model_name
        self.model_asset = hf_hub_download(repo_id=repo_id, filename=model_name, revision="onnx")
        self.model: md.VLM = md.vl(model=self.model_asset)

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
        image_embeds = self.model.encode_image(pil_image)

        results = []
        for i, question in enumerate(questions):
            answer = self.model.query(image_embeds, question)["answer"]
            results.append(answer.strip())

        return results

    @property
    def name(self) -> str:
        return f"Moondream CPU with model {self.model_name}"
