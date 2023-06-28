from dataclasses import dataclass, field
import os
from os.path import isdir, isfile
from pathlib import Path
import sys

from transformers import AutoTokenizer


@dataclass
class GptqConfig:
    ckpt: str = field(
        default=None,
        metadata={
            "help": "Load quantized model. The path to the local GPTQ checkpoint."
        },
    )
    wbits: int = field(default=16, metadata={"help": "#bits to use for quantization"})
    groupsize: int = field(
        default=-1,
        metadata={"help": "Groupsize to use for quantization; default uses full row."},
    )
    act_order: bool = field(
        default=True,
        metadata={"help": "Whether to apply the activation order GPTQ heuristic"},
    )

def load_autogptq_quantized(model_name, device):
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except Exception as err:
        raise ValueError("Please install auto_gptq")
    print(f"loading autogpt model from {model_name}, loading fast tokenizer..")
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              use_fast=True)
    model_basename = None
    for ext in ["*.pt", "*.safetensors"]:
        matched_result = sorted(Path(model_name).glob(ext))
        if len(matched_result) > 0:
            model_basename = matched_result[-1].name
            model_basename = model_basename[:-len(ext[1:])]
            use_safetensors = ext == "*.safetensors"
            print(f"loading model: {model_basename}; use fasettensor: {use_safetensors}")
            break
    if model_basename is None:
        raise ValueError(f"No quantized model found in path {model_name}")
    if device != "cuda:0":
        print(f"TODO: only support single gpu for loading autogptq model now. \
            but get device == {device}, set device to cuda:0"
    )
    device = "cuda:0"
    
    model = AutoGPTQForCausalLM.from_quantized(model_name, 
                                           model_basename=model_basename,
                                           use_safetensors=use_safetensors,
                                           device="cuda:0",
                                           use_triton=False)
    return model, tokenizer 

def load_gptq_quantized(model_name, gptq_config: GptqConfig, device):
    print("Loading GPTQ quantized model...")

    try:
        script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        module_path = os.path.join(script_path, "../repositories/GPTQ-for-LLaMa")

        sys.path.insert(0, module_path)
        from llama import load_quant
    except ImportError as e:
        print(f"Error: Failed to load GPTQ-for-LLaMa. {e}")
        print("See https://github.com/lm-sys/FastChat/blob/main/docs/gptq.md")
        sys.exit(-1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # only `fastest-inference-4bit` branch cares about `act_order`
    if gptq_config.act_order:
        model = load_quant(
            model_name,
            find_gptq_ckpt(gptq_config),
            gptq_config.wbits,
            gptq_config.groupsize,
            act_order=gptq_config.act_order,
        )
    else:
        # other branches
        model = load_quant(
            model_name,
            find_gptq_ckpt(gptq_config),
            gptq_config.wbits,
            gptq_config.groupsize,
        )
    model.to(device)

    return model, tokenizer


def find_gptq_ckpt(gptq_config: GptqConfig):
    if Path(gptq_config.ckpt).is_file():
        return gptq_config.ckpt

    for ext in ["*.pt", "*.safetensors"]:
        matched_result = sorted(Path(gptq_config.ckpt).glob(ext))
        if len(matched_result) > 0:
            return str(matched_result[-1])

    print("Error: gptq checkpoint not found")
    sys.exit(1)
