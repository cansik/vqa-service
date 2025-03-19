from typing import List

import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from vqa.utils import torch_utils


class Florence2Backend:
    def __init__(self, model_name: str = "microsoft/Florence-2-large"):
        self.device = torch_utils.get_device()
        self.torch_dtype = torch_utils.get_dtype(allow_float_16=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model.eval()

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        # Convert the NumPy image to a PIL image
        pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
        results = []

        # Process each question individually
        for question in questions:
            # Prepare the inputs using the processor
            inputs = self.processor(
                text=question,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device, self.torch_dtype)

            # Generate output from the model
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
            # Decode the generated tokens
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=question,
                image_size=(pil_image.width, pil_image.height)
            )
            answer = list(parsed_answer.values())[0]
            results.append(answer.strip())

        return results
