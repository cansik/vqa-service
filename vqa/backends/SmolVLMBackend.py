import re
from typing import List

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from vqa.utils import torch_utils


class SmolVLMBackend:
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        self.device = torch_utils.get_device()
        self.torch_dtype = torch_utils.get_dtype(allow_float_16=True, allow_bfloat_16=True)

        # Select attention implementation based on device.
        attn_impl = "flash_attention_2" if self.device == "cuda" else "eager"

        self.answer_regex = r"^Assistant:(.*).$"

        # Load processor and model with trust_remote_code enabled.
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            _attn_implementation=attn_impl,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        # Convert the input NumPy image to a PIL image.
        pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
        results = []

        # Iterate over each question.
        for question in questions:
            # Construct the conversation message list.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            # Apply the chat template to generate a prompt.
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            # Prepare the inputs (both text and image) for the model.
            inputs = self.processor(text=prompt, images=[pil_image], return_tensors="pt")
            inputs = inputs.to(self.device, self.torch_dtype)

            # Generate the output.
            generated_ids = self.model.generate(**inputs, max_new_tokens=500)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            matches = re.findall(self.answer_regex, generated_text, re.MULTILINE)
            last_answer = "No Answer"
            for match in matches:
                last_answer = match

            results.append(last_answer.strip())

        return results
