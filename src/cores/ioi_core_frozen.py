#!/usr/bin/env python3
"""
ioi_core_frozen.py: runs ioi_core.py unchanged, but stores gradients only for the hidden states.

By default backward() also stores one for every weight, which for Qwen2.5-32B is about 60 GiB on top of the 60 GiB model.
The embedding stays trainable so that the hidden states stay on the graph; freezing the other weights does not change
the gradients with respect to the hidden states. Takes the same arguments as ioi_core.py.
"""
import runpy
from pathlib import Path

from transformers import AutoModelForCausalLM

load = AutoModelForCausalLM.from_pretrained


def frozen(*args, **kwargs):
    model = load(*args, **kwargs)
    for p in model.parameters():
        p.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)
    return model


AutoModelForCausalLM.from_pretrained = frozen
runpy.run_path(str(Path(__file__).with_name("ioi_core.py")), run_name="__main__")
