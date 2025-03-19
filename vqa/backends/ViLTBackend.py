from transformers import ViltProcessor, ViltForQuestionAnswering

from vqa.utils import torch_utils


class ViLTBackend:
    def __init__(self):
        self.device = torch_utils.get_device_string()
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(self.device)

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
