# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# from factscore.utils import convert_model_to_int8_on_gpu
from .lm import LM

class Llama3LM(LM):
    def __init__(self, model_name, model_dir=None, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        if self.model_dir:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to("cuda")
        # self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

if __name__ == "__main__":
    # Set model information
    name = "meta-llama/Llama-3.1-8B"  # Replace with your actual model path if needed
    # model_dir = "/path/to/your/model/directory"  # Set to None if you're using model_name directly

    # Initialize the Llama3LM class
    llama_model = Llama3LM(model_name=name)

    # Load the model and tokenizer
    llama_model.load_model()
    print("Model and tokenizer loaded successfully.")

    # Define a sample prompt for testing the generation method
    test_prompt = "Once upon a time in a faraway kingdom, there was a young prince who"

    # Generate text based on the prompt
    generated_text, scores = llama_model._generate(
        prompts=test_prompt,
        max_sequence_length=2048,
        max_output_length=50,  # Short output for testing
        end_if_newline=True,
        verbose=True
    )

    # Print the generated text and scores
    print("Generated Text:", generated_text)
    print("Generation Scores:", scores)
