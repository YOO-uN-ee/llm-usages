import os
import random
import pandas as pd
import polars as pl

import torch
from transformers import TrainingArguments

from datasets import Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported    # For training model; PEFT

class BasicLLaMA:
    def __init__(self,
                 instruction:str,
                 path_model:str='unsloth/llama-3-8b-bnb-4bit',) -> None:
        
        self.max_seq_length = 2048
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = path_model,
            max_seq_length = self.max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        self.eos_tok = self.tokenizer.eos_token

        # Load lora version of model
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
        
        # TODO: instruction should be required
        self.instruction = instruction
        self.prompt_template = """
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {}

            ### Input:
            {}

            ### Response:
            {}
        """
        self.path_model = path_model

    def data_loading(self,
                     path_file:str):
        try:
            self.data = pl.DataFrame(path_file)
            return 0
        
        except:
            return -1
        
    def convert_to_prompt(self,
                          input_prompt:str,
                          response:str = None,):
        if not response:
            response = ""

        text = self.prompt_template.format(input_prompt, response) + self.eos_tok

        return {"text": text}

    def training(self,
                 training_data: pl.DataFrame,
                 path_saved_model:str='./model'):
        # TODO: Code cleaning
        # TODO: need to convert column to input and response

        df = training_data.sample(fraction=1, shuffle=True)
        df = df.to_pandas()
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda x: self.convert_to_prompt(x['input'], x['response']), 
                              batched = True,)
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 100,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = random.randint(0, 9999),
                output_dir = "outputs",
            ),
        )

        trainer_stats = trainer.train()

        self.path_saved_model = path_saved_model

        self.model.save_pretrained(path_saved_model) # Local saving
        self.tokenizer.save_pretrained(path_saved_model)

    def validating(self,
                   testing_data: pl.DataFrame,):
        
        # Load inference model
        if not self.path_saved_model:
            path_inference_model = self.path_saved_model
        else:
            path_inference_model = self.path_model

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = path_inference_model,
            max_seq_length = self.max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)

        # Load dataframe
        testing_data = testing_data.with_columns(
            predicted_output = pl.struct(pl.col('input_prompt')).map_elements(lambda x: self.convert_to_prompt(x['input_prompt']))
        )

        # TODO: convert to individual input format and fill up the dataframe with the output

        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)

        return tokenizer.batch_decode(outputs)[0]