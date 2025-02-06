import os
import sys
import logging
import json
import argparse
import nltk
from factscore.atomic_facts import AtomicFactGenerator
from metrics.bs import AFGBertScores
from envs import ROOT


os.chdir(sys.path[0])
nltk.download("punkt_tab")


print(f"started: {__file__}")

parser = argparse.ArgumentParser(description="Command-line interface for Atomic Fact Generation.")

parser.add_argument("--hf_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model path")
parser.add_argument("--examples_path", type=str, default="/netscratch/fonseca/OpenFActScore/data/labeled/InstructGPT.jsonl", help="Path to the examples file")
parser.add_argument("--out_dir", type=str, default="results/metrics/afg_bert_score/", help="Output directory")
parser.add_argument("--report_path", type=str, default=None, help="Path to the report file (optional)")

args = parser.parse_args()

HF_MODEL = args.hf_model
EXAMPLES_PATH = args.examples_path
OUT_DIR = args.out_dir
REPORT_PATH = args.report_path

model_name = os.path.basename(HF_MODEL)
out_file = os.path.basename(EXAMPLES_PATH).replace(".jsonl", f"_{model_name}-afs.jsonl")
print(f"Out file: {out_file}")
output_path = os.path.join(ROOT, OUT_DIR, out_file)

print(f"HF Model: {args.hf_model}")
print(f"Examples Path: {args.examples_path}")
print(f"Output Directory: {args.out_dir}")
print(f"Report Path: {args.report_path}")
print(f"Output Path: {output_path}")


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        filename="/netscratch/fonseca/OpenFActScore/afg_llama.log",
                        filemode='w',
                        level=logging.DEBUG)

generator = AtomicFactGenerator(HF_MODEL, ".cache/factscore/demos", af_cache_file="AAAA.pkl")

final_jsonl = []
with open (EXAMPLES_PATH, encoding="utf-8", mode="r") as f:
    for line in f:
        cur_ex = json.loads(line)
        atomic_facts, para_breaks = generator.run(cur_ex["output"])
        try:
            assert len(atomic_facts) == len(cur_ex["annotations"])
        except TypeError:
            logger.debug("Model: %s Atomic Facts: %s\nCurrent_annotations %s\n",
                        HF_MODEL,
                        atomic_facts,
                        cur_ex["annotations"])
            continue
        # cur_ex["annotations"].append({"llama-atomic-facts":
        # [{"text": str}]})
        for atom, annotation in zip(atomic_facts, cur_ex["annotations"]):
            assert atom[0] == annotation["text"]
            annotation['model-atomic-facts'] = [{"text": fact} for fact in atom[1]]

        final_jsonl.append(cur_ex)

logger.debug("Started saving")
with open(output_path, encoding="utf-8", mode="w") as out_f:
    for item in final_jsonl:
        logger.debug("type of item is: %s", type(item))
        logger.debug("item:\n%s", json.dumps(item))
        out_f.write(json.dumps(item) + "\n")

# TODO: Get BertScores for the generations

bs = AFGBertScores(output_path)

bs.get_report(report_path=REPORT_PATH)
