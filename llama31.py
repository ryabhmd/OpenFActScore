import logging
import json
from factscore.factscorer import FactScorer
# from factscore.utils import setup_logger
import os, sys
import numpy as np

os.chdir(sys.path[0])

def convert_to_serializable(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):  # If arrays are involved
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

input_path = "data/labeled/InstructGPT.jsonl"
#model_name="retrieval+llama31"
afv_model = "Llama-3.1-8B"
afg_model = "Llama-3.1-8B-Instruct"
is_npm = True
is_retrieval = True
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
af_annotator = "human-atomic-facts"   #"model-atomic-facts" "human-atomic-facts"



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

# TODO: Add setting name to output dict

logger.critical("FActScore = %.1f%%" % (100*out["score"]))
if "init_score" in out:
    logger.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
logger.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
logger.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

logger.critical(f"{type(out)}")
print(out)

# Save out as a json file
with open(input_path.replace(".jsonl", f"_factscore_output_{af_annotator}.json"), 'w') as f:
    f.write(json.dumps(out, default=convert_to_serializable) + "\n")

#with open("output.json", "w") as f:
#    f.write(json.dumps(out, default=convert_to_serializable) + "\n")
