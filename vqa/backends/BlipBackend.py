from typing import List

import numpy as np
import torch
from transformers import AutoProcessor, BlipForQuestionAnswering

from vqa.backends.BaseVQABackend import BaseVQABackend
from vqa.utils import torch_utils


class BlipBackend(BaseVQABackend):
    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        self.model_name = model_name
        self.device = torch_utils.get_device_string()

        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def process(self, image: np.ndarray, questions: List[str]) -> List[str]:
        # preprocess image
        image = self.processor.image_processor(image, return_tensors="pt").to(self.device)

        # preprocess texts
        questions = [self.processor.tokenizer(text=q, return_tensors="pt").to(self.device) for q in questions]

        with torch.no_grad():
            # compute image embedding
            vision_outputs = self.model.vision_model(pixel_values=image["pixel_values"])
            image_embeds = vision_outputs[0]
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

            answers = []
            for question in questions:
                # compute text encodings
                question_outputs = self.model.text_encoder(
                    input_ids=question["input_ids"],
                    attention_mask=None,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=False,
                )
                question_embeds = question_outputs[0]
                question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(
                    question_embeds.device)
                bos_ids = torch.full(
                    (question_embeds.size(0), 1), fill_value=self.model.decoder_start_token_id,
                    device=question_embeds.device
                )

                outputs = self.model.text_decoder.generate(
                    input_ids=bos_ids,
                    eos_token_id=self.model.config.text_config.sep_token_id,
                    pad_token_id=self.model.config.text_config.pad_token_id,
                    encoder_hidden_states=question_embeds,
                    encoder_attention_mask=question_attention_mask,
                )

                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                answers.append(answer)

            return answers

    @property
    def name(self) -> str:
        return f"Blip with model {self.model_name}"
