# pip install namo

import io
from typing import List

import numpy as np
import requests
from PIL import Image

from vqa.backends.BaseVQABackend import BaseVQABackend
from vqa.utils import torch_utils


def load_image(image_file):
    if isinstance(image_file, Image.Image):
        return image_file
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


class NamoBackend(BaseVQABackend):
    def __init__(self):
        # monkeypatch namo utils
        # https://github.com/lucasjinreal/Namo-R1/blob/main/namo/utils/infer_utils.py#L16-L17
        import namo.utils.infer_utils as infer_utils
        infer_utils.load_image = load_image

        from namo.api.vl import VLInfer
        device_str = torch_utils.get_device_string()
        self.model: VLInfer = VLInfer(model_type="namo", device=device_str)

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        pil_image = Image.fromarray(np.uint8(image)).convert("RGB")

        results = []
        for i, question in enumerate(questions):
            answer = self.model.model.generate(question, [pil_image], stream=False, keep_history=False)
            results.append(answer.strip())

        return results

    @property
    def name(self) -> str:
        return f"Namo"
