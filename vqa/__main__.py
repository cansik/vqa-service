import argparse
import datetime
import random
import time
from typing import Optional

from gradio.processing_utils import PUBLIC_HOSTNAME_WHITELIST

PUBLIC_HOSTNAME_WHITELIST.append("localhost")
PUBLIC_HOSTNAME_WHITELIST.append("127.0.0.1")

import gradio
import numpy as np
from rich_argparse import ArgumentDefaultsRichHelpFormatter

from vqa.backends import VQA_Backends
from vqa.backends.BaseVQABackend import BaseVQABackend
from vqa.utils.argparse_utils import add_dict_choice_argument

vqa_backend: Optional[BaseVQABackend] = None


def process(image: np.ndarray, questions_text: str):
    questions = questions_text.split("\n")
    start = time.perf_counter()
    answers = vqa_backend.process(image, questions)
    end = time.perf_counter()

    delta = end - start

    print(f"{datetime.datetime.now()}: Processed image in {delta:.3f}s")

    return {"questions": questions, "answers": answers}, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vqa", description="VQA Service",
                                     formatter_class=ArgumentDefaultsRichHelpFormatter)

    parser.add_argument("--host", type=str, default="127.0.0.1", help="Service host")
    parser.add_argument("--port", type=int, default=7840, help="Service port")
    add_dict_choice_argument(parser, VQA_Backends, "--backend", help="VQA Backend", default="blip")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"loading vqa model {str(args.backend)}...")
    global vqa_backend
    vqa_backend = args.backend()

    print("starting service...")
    interface = gradio.Interface(fn=process,
                                 title="Visual Question and Answer",
                                 description=f"Using {vqa_backend.name}",
                                 inputs=[gradio.Image(), gradio.Textbox(label="Questions (one per line)", lines=5)],
                                 outputs=[gradio.Json(label="Answers"), gradio.Number(label="Inference Time")],
                                 flagging_mode="never",
                                 analytics_enabled=False,
                                 cache_examples=True,
                                 examples=[
                                     [
                                         f"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png?{random.random()}",
                                         "Is this a Bus?"
                                     ]
                                 ],
                                 delete_cache=(5, 1)
                                 )
    interface.launch(share=False, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
