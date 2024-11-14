import logging
import json
from factscore.factscorer import FactScorer

input_path = "data/labeled/InstructGPT.jsonl"
model_name="retrieval+llama31+npm"
gamma=10
openai_key="api.key"
data_dir=".cache/factscore/"
model_dir = ".cache/factscore/"
cache_dir = ".cache/factscore/"
knowledge_source = None
cost_estimate = "consider_cache"
abstain_detection_type = None
use_atomic_facts = True
verbose = False
print_rate_limit_error = False
n_samples = None


logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if print_rate_limit_error else logging.CRITICAL)

fs = FactScorer(model_name=model_name,
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
            atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
        else:
            topics.append(dp["topic"])
            generations.append(dp["output"])
        if n_samples is not None and tot==n_samples:
            break
out = fs.get_score(topics=topics,
                   generations=generations,
                   gamma=gamma,
                   atomic_facts=atomic_facts if use_atomic_facts else None,
                   knowledge_source=knowledge_source,
                   verbose=verbose)
logging.critical("FActScore = %.1f%%" % (100*out["score"]))
if "init_score" in out:
    logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

# Save out as a json file
with open(input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
    f.write(json.dumps(out) + "\n")
