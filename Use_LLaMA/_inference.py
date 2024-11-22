import torch
import pandas as pd
import polars as pl
import os
from unsloth import FastLanguageModel

streetname_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the given streetname into one of the categories. Base it on the streetname etymology.

### Input:
{} located in {}, {}

### Response:
{}
"""

def running(struct_input:dict, model_path:str) -> str:
    # INFERENCE
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    global EOS_TOKEN 
    EOS_TOKEN = tokenizer.eos_token

    inputs = tokenizer(
    [
        streetname_prompt.format(
            struct_input["streetname_full"],
            struct_input["PropertyCity"],
            struct_input["state"],
            "",
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)

    answers = tokenizer.batch_decode(outputs)[0]
    
    return answers

def validate(file_name:str, save_path:str, model_path:str|None=None):
    df = pl.read_csv(file_name)
    df = df.with_columns(
        llama_estimate = pl.struct(pl.col(['streetname_full','PropertyCity','state'])).map_elements(lambda x: running(struct_input=x, model_path=model_path))
    )

    df.write_csv(save_path)