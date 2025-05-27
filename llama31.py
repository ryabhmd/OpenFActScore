import logging
import json
from factscore.factscorer import FactScorer
import os, sys
import numpy as np
from metrics.ck import CKScore
import argparse

os.chdir(sys.path[0])

def convert_to_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):  # If arrays are involved
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="CLI for configuring fact scoring tool")

    parser.add_argument("--input_path", type=str, default="data/labeled/InstructGPT.jsonl", help="Path to the input data file")
    parser.add_argument("--afv_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path or name of the AFV model")
    parser.add_argument("--afg_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path or name of the AFG model")
    parser.add_argument("--is_npm", action="store_true", help="Enable NPM mode")
    parser.add_argument("--is_retrieval", action="store_true", default=True, help="Enable retrieval mode")
    parser.add_argument("--gamma", type=int, default=10, help="Gamma value for scoring")
    parser.add_argument("--openai_key", type=str, default="api.key", help="OpenAI API key")
    parser.add_argument("--data_dir", type=str, default=".cache/factscore/", help="Directory for data cache")
    parser.add_argument("--model_dir", type=str, default=".cache/factscore/", help="Directory for model cache")
    parser.add_argument("--cache_dir", type=str, default=".cache/factscore/", help="Directory for cache files")
    parser.add_argument("--knowledge_source", type=str, default=None, help="Path to the knowledge source")
    parser.add_argument("--cost_estimate", type=str, default="consider_cache", help="Strategy for cost estimation")
    parser.add_argument("--abstain_detection_type", type=str, default=None, help="Type of abstain detection")
    parser.add_argument("--use_atomic_facts", action="store_true", help="Enable use of atomic facts")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")
    parser.add_argument("--print_rate_limit_error", action="store_true", default=False, help="Print rate limit errors")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--af_annotator", type=str, default="human-atomic-facts", help="Annotator type (e.g., model-atomic-facts, human-atomic-facts)")

    args = parser.parse_args()

    input_path = args.input_path
    afv_model = args.afv_model # "meta-llama/Llama-3.1-8B-Instruct" # "mistralai/Mistral-7B-Instruct-v0.1" #
    afg_model = args.afg_model #"meta-llama/Llama-3.1-8B-Instruct"
    is_npm = args.is_npm
    is_retrieval = args.is_retrieval
    gamma = args.gamma
    openai_key = args.openai_key# "api.key"
    data_dir = args.data_dir #".cache/factscore/"
    model_dir = args.model_dir # ".cache/factscore/"
    cache_dir = args. cache_dir #".cache/factscore/"
    knowledge_source = args.knowledge_source # None
    cost_estimate = args.cost_estimate # "consider_cache"
    abstain_detection_type = args.abstain_detection_type # None
    use_atomic_facts = args.use_atomic_facts # True
    verbose = args.verbose # False
    print_rate_limit_error = args.print_rate_limit_error #False
    n_samples = args.n_samples # None
    af_annotator = args.af_annotator # "human-atomic-facts"   #"model-atomic-facts" "human-atomic-facts"



    print(f"started: {__file__}")
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            filename="/netscratch/fonseca/OpenFActScore/llama31.log",
                            filemode='a',
                            level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.debug("Started logger: Calling FactScorer")

    print("calling FactScore")
    fs = FactScorer(afv_model=afv_model,
                    afg_model = afg_model,
                    is_npm = is_npm,
                    is_retrieval = is_retrieval,
                    data_dir=data_dir,
                    model_dir=model_dir,
                    cache_dir=cache_dir,
                    openai_key=openai_key,
                    cost_estimate=cost_estimate,
                    abstain_detection_type=abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] if sent[af_annotator] for atom in sent[af_annotator]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if n_samples is not None and tot==n_samples:
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

    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=gamma,
                       atomic_facts=atomic_facts if use_atomic_facts else None,
                       knowledge_source=knowledge_source,
                       verbose=verbose)


    logger.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logger.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logger.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logger.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Save out as a json file
    out_path = input_path.replace(".jsonl", f"_fs_{af_annotator}{afv_model.split('/')[-1]}.json")
    with open (out_path, 'w') as f:
        f.write(json.dumps(out, default=convert_to_serializable) + "\n")
    print(f"Saved to: {out_path}")

    if use_atomic_facts and af_annotator == "human-atomic-facts":
        # Get CK agreement
        ck_scorer = CKScore(out_path)
        out["cohen_kappa"] = ck_scorer.get_cks()

        logger.critical("Cohen's kappa: %.2f" % out["cohen_kappa"])
        logger.critical("Saved to: %s", out_path)
        print(f"Saved CK Score to: {out_path}")

        with open (out_path, 'w') as f:
            f.write(json.dumps(out, default=convert_to_serializable) + "\n")
