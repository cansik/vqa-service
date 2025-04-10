from functools import partial

from vqa.backends.Blip2Backend import Blip2Backend
from vqa.backends.BlipBackend import BlipBackend
from vqa.backends.MoondreamBackend import MoondreamBackend
from vqa.backends.MoondreamCpuBackend import MoondreamCpuBackend
from vqa.backends.SmolVLMBackend import SmolVLMBackend
from vqa.backends.ViLTBackend import ViLTBackend

VQA_Backends = {
    "blip": BlipBackend,
    "blip2": Blip2Backend,
    "blip2-flan": partial(Blip2Backend, "Salesforce/blip2-flan-t5-xl"),
    "vilt": ViLTBackend,
    "moondream": MoondreamBackend,
    "moondream-cpu": MoondreamCpuBackend,
    "smolvlm": SmolVLMBackend,
    "smolvlm2": partial(SmolVLMBackend, "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
}
