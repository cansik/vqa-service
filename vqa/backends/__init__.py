from functools import partial

from vqa.backends.Blip2Backend import Blip2Backend
from vqa.backends.BlipBackend import BlipBackend
from vqa.backends.MoondreamBackend import MoondreamBackend
from vqa.backends.MoondreamCpuBackend import MoondreamCpuBackend
from vqa.backends.NamoBackend import NamoBackend
from vqa.backends.SmolVLMBackend import SmolVLMBackend
from vqa.backends.VLMForMLXBackend import VLMForMLXBackend
from vqa.backends.ViLTBackend import ViLTBackend

VQA_Backends = {
    "blip": BlipBackend,
    "blip2": Blip2Backend,
    "blip2-flan": partial(Blip2Backend, "Salesforce/blip2-flan-t5-xl"),
    "vilt": ViLTBackend,
    "vlmmlx": VLMForMLXBackend,
    "vlmmlx-phi35": partial(VLMForMLXBackend, "mlx-community/Phi-3.5-vision-instruct-4bit"),
    "vlmmlx-smolvlm2": partial(VLMForMLXBackend, "mlx-community/SmolVLM2-500M-Video-Instruct-mlx-8bit-skip-vision"),
    "namo": NamoBackend,
    "moondream": MoondreamBackend,
    "moondream-cpu": MoondreamCpuBackend,
    "smolvlm": SmolVLMBackend,
    "smolvlm2": partial(SmolVLMBackend, "HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
}
