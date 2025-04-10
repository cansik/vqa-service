from transformers import ViltProcessor, ViltForQuestionAnswering

from vqa.utils import torch_utils


class ViLTBackend:
    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa"):
        self.model_name = model_name
        self.device = torch_utils.get_device_string()
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(self.device)

    def process(self, image, questions):
        results = []
        for i, q in enumerate(questions):
            # prepare inputs
            encoding = self.processor(image, q, return_tensors="pt").to(self.device)

            # forward pass
            outputs = self.model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            result = self.model.config.id2label[idx]
            results.append(result)

        return results

    @property
    def name(self) -> str:
        return f"ViLT with model {self.model_name}"
