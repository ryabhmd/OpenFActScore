import argparse
import string
import json
import logging
import os
import numpy as np
import torch
import gc

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.Llama3LM import Llama3LM
from factscore.HFmodel import HFmodel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):
    def __init__(self,
                 tokenizer,
                 afv_model="meta-llama/Llama-3.1-8B-Instruct",
                 afg_model="meta-llama/Llama-3.1-8B-Instruct",
                 is_npm=False,
                 is_logits=True,
                 #is_retrieval=True,
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        self.tokenizer = tokenizer
        self.afg_model = afg_model
        self.afv_model = afv_model
        self.is_npm = is_npm
        self.model_name = self.generate_config_name()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type
        self.is_logits = is_logits

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate
        if "inst-llama" in self.model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in self.model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        elif "Llama-3.1" in self.afv_model:
            self.lm = Llama3LM(self.afv_model,
                                cache_file=os.path.join(cache_dir, self.model_name))
        else:
            self.lm = HFmodel(self.afv_model,
                                cache_file=os.path.join(cache_dir, self.model_name),
                                logits=self.is_logits)
            self.lm.load_model()
            
        self.logger.debug("%s",self.model_name)

    def generate_config_name(self):
        afg_name = self.afg_model.split('/')[-1]
        afv_name = self.afv_model.split('/')[-1]
        model_name = [afg_name, afv_name]
        if self.is_npm:
            model_name.append("npm")
        model_name.insert(0,"retrieval")
        model_name = "+".join(model_name)
        return model_name

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, encoder, name="enwiki-20230401", db_path=None, data_path=None, retrieval_type = "gtr-t5-large"):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        print(f"db_path:{db_path}")
        print(f"data_path:{data_path}")
        self.db[name] = DocDB(
                tokenizer=self.tokenizer, 
                db_path=db_path, 
                data_path=data_path
                )

        self.retrieval[name] = Retrieval(
                encoder=encoder, 
                db=self.db[name], 
                cache_path=cache_path, 
                embed_cache_path=embed_cache_path, 
                retrieval_type=retrieval_type, 
                batch_size=self.batch_size
                )

        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        # Davinci-003 discontinued
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.0015

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("""Estimated OpenAI API cost for %s ($%.3f per 1000 tokens):
        $%.2f for %d words and %d tokens",task, rate, total_cost, total_words, total_tokens""")

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)
            print(f"Registered knowledge source: {knowledge_source}.")

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        ## I can provide the Atomic Facts myself and by pass the AF generation if I want to test 
        ## the evaluation
        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else: #Atomic FactGeneration
            if self.af_generator is None:
                print("Generating atomic facts.")
                self.af_generator = AtomicFactGenerator(model_name=self.afg_model,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        key_path=self.openai_key,
                                                        af_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            if "ChatGPT" in self.model_name:
                total_words = 0
                for gen in generations:
                    total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

                self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

            if verbose:
                topics = tqdm(topics)

            ## Start obtaining Atomic Facts for each generation 
            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                print("Response is not abstained.")
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            print(f"Generated atomic facts: {atomic_facts}.")
            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()
            self.af_generator.lm.unload_model()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts[0]])
        print(f"Respond ratio is: {respond_ratio}.")

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        # 
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            print(f"Topic: {topic}.")
            print(f"Generation: {generation}.")
            print(f"Facts: {facts}.")
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                print(f"Knowledge source: {knowledge_source}.")
                print(f"Decision: {decision}.")
                if decision is None:
                    decisions.append(None)
                    continue
                # Score is the average number of "is_supported" for generation
                score = np.mean([d["is_supported"] for d in decision])
                print(f"Score: {score}.")
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None]),
               "config_name": self.model_name}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _chunked(self, iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]


    def _get_score_batches(
            self,
            topic,
            generation,
            atomic_facts,
            knowledge_source,
            batch_size=4,
            gamma=10,
        ):
        """
        The function is designed to be used per entity -> returnd dict of decisions for each atom in entity + the overall OFS of the entity
        """
        decisions = []
        total_words = 0

        prompts = []
        atoms = []
        contexts = []

        # 1. Build prompts for all atomic facts in topic
        
        # get all passages for topic -> this is the same for all atoms so no need to do it in the loop

        all_passages = self.retrieval[knowledge_source].db.get_paragraphs_from_db()

        for atom in atomic_facts[0]:
            
            atom = atom.strip()

            passages = self.retrieval[knowledge_source].get_passages(topic[0], atom, k=5, all_passages=all_passages)

            definition = f"Answer the question about {topic} based on the given context.\n\n"

            # build context from passages
            context = ""
            for psg in reversed(passages):
                context += f"Text: {psg.replace('<s>', '').replace('</s>', '')}\n\n"

            definition += context.strip()
            if definition[-1] not in string.punctuation:
                definition += "."

            prompt = (
                    f"{definition}\n\n"
                    f"Input: {atom} True or False?\n"
                    f"Answer:"
                 )

            prompts.append(prompt)
            atoms.append(atom)
            contexts.append(context)

        print(f"Built all prompts for topic: {topic[0]}")
        self.logger.critical(f"Built all prompts for topic: {topic[0]}") 

        # 2. AFV using batched generation
        if self.lm:

            generations = []
            for batch_prompts in self._chunked(prompts, batch_size):
                with torch.no_grad():
                    outputs = self.lm._generate_batches(batch_prompts)
                generations.extend(outputs)

                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"Received LM generations for {len(prompts)} prompts")
            self.logger.critical(f"Received LM generations for {len(prompts)} prompts")
        else:
            raise RuntimeError(
                    "No LM defined for AFV.")

        # 3. Parse generations to get decisions
        for atom, context, gen in zip(atoms, contexts, generations):
            generated_answer = gen.lower()
            print(f"Generated answer for {atom}: {generated_answer}")

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")

            else:
                tokens = generated_answer.translate(
                        str.maketrans("", "", string.punctuation)
                    ).split()
                is_supported = all(
                        keyword not in tokens
                        for keyword in ["not", "cannot", "unknown", "information"]
                    )

            decisions.append(
                    {
                        "atom": atom,
                        "context": context,
                        "generation": gen,
                        "is_supported": is_supported
                }
            )

        # 4. Compute score for entity
        # Score is the average number of "is_supported" for generation
        init_score = np.mean([d["is_supported"] for d in decisions])
        gamma_score = None

        if gamma and len(atomic_facts[0]) > 0:
            penalty = 1.0 if len(atomic_facts[0])>gamma else np.exp(1-gamma/len(atomic_facts[0]))
            gamma_score = penalty * init_score

        return decisions, init_score, gamma_score



    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        
        # Prompt Construction
        for atom in atomic_facts:
            atom = atom.strip()
            print(f"Atom: {atom}.")
            if self.lm:
                self.logger.debug("Retrieving passages from %s", knowledge_source)
                print("Retrieving passages from %s", knowledge_source)
                print(f"Retrieval object: {self.retrieval}")
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                
                if not passages:
                    return None
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Text: {}\n\n".format(psg.replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if definition[-1] not in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nAnswer:".format(definition.strip(), atom.strip())
                print(f"Prompt: {prompt}")

                output = self.lm._generate(prompt)
                print(f"Output: {output}")

                if isinstance(output[1], np.ndarray) and self.lm.logits:
                    # TODO: assert with tokenizer vocab len
                    logits = np.array(output[1])
                    # when logits are available,
                    true_score = logits[self.lm.true_id]
                    false_score = logits[self.lm.false_id]
                    is_supported = true_score > false_score
                    is_supported = is_supported.item()
                    self.logger.debug("-------------------")
                    self.logger.debug(f"Prompt: {prompt}")
                    self.logger.debug(f'\nLogits:\nTrue: {true_score}\nFalse: {false_score}\nis_supported: {is_supported}')
                    self.logger.debug(f'Output: {output[0]}')
                    self.logger.debug("-------------------")
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    print(f"generated answer: {generated_answer}")
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append(
                {
                    "atom": atom, 
                    "retrieval_results": context, 
                    #"afv_model_output": generated_answer, 
                    "is_supported": is_supported
                    }
                )
        if cost_estimate:
            return total_words
        else:
            return decisions

def convert_to_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):  # If arrays are involved
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute FactScore for generated outputs.")

    # Required arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the input JSONL file containing topics and generations.")

    # Model configuration arguments
    parser.add_argument('--afv_model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the Atomic Fact Verification model.")
    parser.add_argument('--afg_model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Name of the Atomic Fact Generation model.")
    parser.add_argument('--is_npm', action='store_true',
                        help="Flag to enable Non Parametric Model (NPM).")

    # Directories and paths
    parser.add_argument('--data_dir', type=str, default=".cache/factscore",
                        help="Directory to store data files.")
    parser.add_argument('--model_dir', type=str, default=".cache/factscore",
                        help="Directory to store model files.")
    parser.add_argument('--cache_dir', type=str, default=".cache/factscore",
                        help="Directory to store cache files.")
    parser.add_argument('--openai_key', type=str, default="api.key",
                        help="Path to the OpenAI API key file.")

    # Evaluation configuration
    parser.add_argument('--gamma', type=int, default=10,
                        help="Hyperparameter for length penalty.")
    parser.add_argument('--knowledge_source', type=str, default=None,
                        help="Name of the knowledge source for retrieval (the .jsonl file).")
    # Added the argument below in case I already registered my knowledge sources, in this scenario I can give it the path to the DB file. 
    parser.add_argument('--knowledge_source_path', type=str, default=None,
                        help="Path of the knowledge source (formatted as .db) for retrieval.")
    # Added the argument below to register the name of the DB.
    parser.add_argument('--knowledge_source_name', type=str, default=None,
                        help="Name of the registered DB (to be used in cache when saving results).")
    parser.add_argument('--cost_estimate', type=str, default="consider_cache",
                        choices=["consider_cache", "ignore_cache"],
                        help="Option to consider or ignore cache in cost estimation.")
    parser.add_argument('--abstain_detection_type', type=str, default=None,
                        choices=["perplexity_ai", "generic", "none"],
                        help="Type of abstain detection to use.")

    # Optional settings
    parser.add_argument('--use_atomic_facts', action='store_true',
                        help="Flag to use pre-existing atomic facts in the input data.")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose mode with progress bars.")
    parser.add_argument('--print_rate_limit_error', action='store_true',
                        help="Print rate limit errors when using OpenAI keys.")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Limit the number of samples to process.")
    parser.add_argument('--debug_logger', action='store_true',
                        help="Set logger level to debug")
    parser.add_argument("--af_annotator", type=str, default="human-atomic-facts",
                        help="Annotator type (e.g., model-atomic-facts, human-atomic-facts)")


    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename=os.path.join(os.getcwd(), __file__.replace(".py",".log")),
                        level=logging.DEBUG if args.debug_logger else logging.CRITICAL)

    logger = logging.getLogger(__name__)
    logger.debug("Started logger: Calling FactScorer")
    # Initialize FactScorer with parsed arguments
    fs = FactScorer(
        afv_model=args.afv_model,
        afg_model=args.afg_model,
        is_npm=args.is_npm,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        openai_key=args.openai_key,
        cost_estimate=args.cost_estimate,
        abstain_detection_type=args.abstain_detection_type,
    )

    # knwoledge source generation
    knwoledge_source = args.knowledge_source
    knowledge_source_name = args.knowledge_source_name
    fs.register_knowledge_source(name=knowledge_source_name, data_path=knwoledge_source)

    print(f"Registered knwoledge source {knowledge_source_name}.")

    topics, generations, atomic_facts = [], [], []
    tot = 0
    logger.critical("Initialized FactScore")
    # Read input file
    with open(args.input_path, 'r', encoding='utf8') as f:
        print(f"Reading {args.input_path}.")
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                topics.append(dp["topic"])
                generations.append(dp["generation"])
                atomic_facts.append(dp["atomic_facts"])
            else:
                topics.append(dp["topic"])
                generations.append(dp["generation"])
                print(f"Extracted topic: {topics}. Atomic facts are not in the input file!")
            if args.n_samples is not None and tot == args.n_samples:
                break
    if atomic_facts:
        filtered_data = [
            (l1, l2, lol)
            for l1, l2, lol in zip(topics, generations, atomic_facts)
            if lol
        ]
        topics, generations, atomic_facts = zip(*filtered_data)
        # Convert tuples back to lists if needed
        topics, generations, atomic_facts = list(topics), list(generations), list(atomic_facts)
    logger.debug("Preparing to get scores")
    # Compute scores
    print("Starting to compute score")
    results = fs.get_score(
        topics=topics,
        generations=generations,
        gamma=args.gamma,
        atomic_facts=atomic_facts if args.use_atomic_facts else None,
        knowledge_source=knowledge_source_name,
        verbose=args.verbose
    )

    # Log results
    logging.critical("FactScore = %.1f%%", (100 * results["score"]))
    if "init_score" in results:
        logging.critical("FactScore w/o length penalty = %.1f%%", (100 * results["init_score"]))
    logging.critical("Respond ratio = %.1f%%", (100 * results["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f", results["num_facts_per_response"])

    # Save results to output file
    out_path = args.input_path.replace(".jsonl", f"_fs_{args.af_annotator}{args.afv_model.split('/')[-1]}.json")
    with open (out_path, 'w') as f:
        f.write(json.dumps(results, default=convert_to_serializable) + "\n")
    print(f"Saved to: {out_path}")

