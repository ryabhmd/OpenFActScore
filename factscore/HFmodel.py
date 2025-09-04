import logging
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from .utils import LLAMA_3_INSTRUCT_TEMPLATE

from .lm import LM

class HFmodel(LM):
    def __init__(self,
                 model_name,
                 cache_file=None,
                 mode="afv",
                 logits=False,
                 use_chat=True):
        if mode not in {"afv","afg"}:
            raise ValueError(f"allowed modes are afg, afv. Not {mode}")
        self.mode = mode
        self.model_name = model_name
        if cache_file:
            super().__init__(cache_file)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logits = logits
        self.use_chat= use_chat

    def load_model(self):
        print("loading model")
        # Load model from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to("cuda")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.logger.debug(f"Loaded model name: {self.model.config._name_or_path}")

        # setting pad token as end of sentence token
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        # Check for chat_template
        # Defining Chat_template
        if "chat_template" in self.tokenizer.init_kwargs and self.use_chat:
            self.use_chat = True
        else:
            self.use_chat = False

        if self.mode=="afv" and self.logits:
            # Is it possible to use logits
            self.true_id = self.tokenizer.convert_tokens_to_ids(["True"])
            self.false_id = self.tokenizer.convert_tokens_to_ids(["False"])
            if len(self.true_id) > 2:
                self.logits = False
                print("Not possible to use logits")
            if len(self.true_id) == 2:
                self.logits = True
                self.true_id = [self.true_id[-1]]
                self.false_id = [self.false_id[-1]]
            else:
                self.logits = True
        else:
            self.logits = False



    def unload_model(self):
        """
        Unloads the model and clears the memory.
        """
        self.model = None           # Delete the model object
        gc.collect()
        torch.cuda.empty_cache()    # Free up GPU memory
        self.logger.debug("Model unloaded and GPU memory cleared.")


    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]
        if self.logits is False and self.mode == "afv":
            max_output_length = 3
        # Force output to the ids for True or False
        force_id = [self.true_id, self.false_id] if self.logits else None

        if self.use_chat:
            prompts = self.chat_formatter(prompts)
        tokens = self.tokenizer(prompts)
        input_ids = tokens.input_ids
        attention_masks = tokens.attention_mask
        if verbose:
            input_ids = tqdm(input_ids)

        generations = []
        scores = []
        for curr_input_ids, attention_mask in zip(input_ids, attention_masks):
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            attention_mask = torch.LongTensor([attention_mask]).cuda()
            gen_outputs = self.model.generate(
                curr_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_output_length,
                return_dict_in_generate=True,
                force_words_ids=force_id,
                output_scores=True,
                num_beams=2 if force_id else 1,  # Use beam search when force_words_ids is specified
                do_sample=False if force_id else True,
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:], skip_special_tokens=True)

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            self.logger.debug("scores: %s\ntokens:%s\ngen:%s", gen_scores, gen_tokens, gen)
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

    def chat_formatter(self, prompts:list):
        """
        Apply the chat formatter and include system prompt
        Formatted_prompts: list
        """
        formatted_prompts = []

        for prompt in prompts:
            if self.mode=="afv":
                system_instruct = "You are an annotator that verifies the factuality of a sentence according to a given source text. You answer only True or False and provide no further explanations."
            else:
                _instruct = """
                You are an annotator that breaks down sentences into independent facts, short statements that each contain one piece of information contained in the given sentence.
                in the next paragraphs you have examples of sentences broken down in atomic facts. 
                You have to complete the example given by the user.
                Do not add new entities, do not deviate from the subject of the sentence given by the user, do not hallucinate, do not repeat facts in the system prompt.
                List the sentences using -
                """
                
                parts = prompt.rsplit("\n\n")
                system_instruct = f"{_instruct}\n{parts[0]}"
                prompt = parts[1]

            instruct_dict = [{"role" : "assistant", "content": system_instruct},
                             {"role": "user", "content": prompt}]

            cur_prompt = self.tokenizer.apply_chat_template(instruct_dict, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(cur_prompt)
            self.logger.debug("After Formatter prompt: %s", cur_prompt)
        return formatted_prompts

if __name__ == "__main__":
    # Set model information
    name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your actual model path if needed

    # Initialize the Llama3LM class
    llama_model = HFmodel(model_name=name)

    # Load the model and tokenizer
    llama_model.load_model()
    print("Model and tokenizer loaded successfully.")

    # Define a sample prompt for testing the generation method
    test_list = []
    test_prompt = "You are a helpful assistant, an oracle of knowledge and output only factual knowledge that answers only True or False.\nThe sky is blue. True or False?"

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
