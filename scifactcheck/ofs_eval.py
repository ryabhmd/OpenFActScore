"""
OpenFactScore Entity-wise Evaluation Pipeline.

This script implements an entity-level factuality evaluation pipeline based on
OpenFactScore (OFS) for scientific text generation.

The pipeline performs the following steps:

1. Load extracted claims produced VeriScore, where each line corresponds to one scientific entity.
2. Convert extracted claims into OpenFactScore-compatible inputs consisting of:
   - topic
   - generation
   - atomic facts
3. Load the SciFactCheck HuggingFace dataset, which provides the list of
   scientific entities and their identifiers.
4. Iterate over entities one by one and, for each entity:
   a. Instantiate a FactScorer object.
   b. Register an entity-specific knowledge source (JSONL), consisting of parsed full-length articles corresponding to each entity.
   c. Compute OpenFactScore using atomic-fact verification.
5. Store detailed, per-entity evaluation outputs including:
   - OpenFactScore results
   - atomic facts
   - retrieval source paths
   - model metadata

The final output is written as a JSONL file where each line corresponds to the
evaluation results of a single scientific entity.
"""

import os
import json
import logging
import argparse
from tqdm import tqdm

from typing import Dict, List, Any, Iterable
from datetime import datetime

from datasets import load_dataset

from scifactcheck.utils import build_openfactscore_input, get_knowledge_source_name
from factscore.factscorer import FactScorer

class OFSEval:
    """
    OpenFactScore-based evaluation pipeline.
    """

    def __init__(
        self,
        extracted_claims_path: str,
        knowledge_source_dir: str, 
        hf_dataset: str = "rabuahmad/scifactcheck",
        afv_model: str = "google/gemma-7b-it",
        data_dir=".cache/factscore",
        model_dir=".cache/factscore",
        cache_dir=".cache/factscore",
        debug_logger=True
    ):
        self.extracted_claims_path = extracted_claims_path
        self.knowledge_source_dir = knowledge_source_dir
        self.afv_model = afv_model
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.debug_logger = debug_logger
        
        # step 1: convert extracted claims file to valid OFS input file
        with open(self.extracted_claims_path, "r") as f:
            self.extracted_claims = [json.loads(line) for line in f]
        self.ofs_input = self.prepare_ofs_inputs(self.extracted_claims)

        # step 2: read HF data
        self.scifactcheck_data = load_dataset(hf_dataset)
        
        print(f"input claims {len(self.ofs_input)}")
        print(f"hf {len(self.scifactcheck_data['train'])}")

        assert len(self.ofs_input) == len(self.scifactcheck_data["train"])
        assert isinstance(self.ofs_input[0], dict)

    def run(
            self, 
            output_file: str
    ):
        """
        Run OpenFactScore evaluation by iterating over scientific entities.
        The function writes OFS results as it iterates over entities in the given output_file path.
        """
        # set up logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            filename=os.path.join(
                os.getcwd(),
                __file__.replace(".py", ".log")
            ),
            level=logging.DEBUG if getattr(self, "debug_logger", False) else logging.CRITICAL
        )

        logger = logging.getLogger(__name__)
        logger.critical("Started OpenFactScore entity-wise evaluation")

        all_results = []

        # step 3: iterate over scientific entities
        # open output file once
        with open(output_file, "w", encoding="utf-8") as f:

            for idx, (entity, ofs_dp) in enumerate(
                tqdm(
                    zip(self.scifactcheck_data["train"], self.ofs_input),
                    total=len(self.ofs_input),
                    desc="OFS for entities"
                )
            ):
                logger.debug(f"Evaluating entity {idx}, {entity}")

                # step 4 initialize FactScorer object
                fs = FactScorer(
                    afv_model=self.afv_model,
                    data_dir=self.data_dir,
                    model_dir=self.model_dir,
                    cache_dir=self.cache_dir,
                )

                logger.debug("Initialized FactScorer")

                # step 5: register knowledge source 
                if entity["paper_doi"]:
                    source = "doi"
                    entity_id = entity["paper_doi"]
                else:
                    source = "iep"
                    entity_id = entity["title"]

                knowledge_source_name = get_knowledge_source_name(entity_idx=idx, source=source, id=entity_id)
                knowledge_source_path = os.path.join(
                    self.knowledge_source_dir,
                    f"{knowledge_source_name}.jsonl"
                    )

                fs.register_knowledge_source(
                    name=knowledge_source_name,
                    data_path=knowledge_source_path,
                )

                logger.debug(
                    f"Registered knowledge source {knowledge_source_name}"
                )

                # step 6: prepare inputs for scoring
                topics = [ofs_dp["topic"]]
                generations = [ofs_dp["generation"]]
                atomic_facts = [ofs_dp["atomic_facts"]]

                # step 7: run FactScorer 
                results = fs.get_score(
                    topics=topics,
                    generations=generations,
                    atomic_facts=atomic_facts,
                    knowledge_source=knowledge_source_name,
                )

                # step 8: store per-entity result 
                entity_result = {
                    "entity_index": idx,
                    "entity": entity["scientific_entity"],
                    "topic": ofs_dp["topic"],
                    "generation": ofs_dp["generation"],
                    "atomic_facts": ofs_dp["atomic_facts"],
                    "ofs_result": results,
                    "knowledge_source": knowledge_source_path,
                    "afv_model": self.afv_model,
                }

                # write results of entity into output file
                f.write(json.dumps(entity_result, ensure_ascii=False) + "\n")
                f.flush()          # ensures progress is saved to disk
                os.fsync(f.fileno())  # extra safety on clusters


        logger.critical("Finished OpenFactScore evaluation")

    
    def prepare_ofs_inputs(
        self,
        entities: Iterable[Dict],
    ) -> List[Dict]:
        """
        Step 1 of the evaluation pipeline.

        Parameters
        ----------
        entities : iterable of dict
            Extracted-claims entities (one per LLM output).

        Returns
        -------
        List[dict]
            OpenFactScore-compatible samples:
            {
              "topic": str,
              "generation": str,
              "atomic_facts": List[str]
            }
        The list should contain 2500 dictionaries corresponding to each scientific concept. 
        """

        samples: List[Dict] = []

        for entity in entities:
            sample = build_openfactscore_input(
                entity,
                drop_empty=False,
                deduplicate=True,
            )

            if sample is not None:
                samples.append(sample)

        return samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_claims_path', type=str, required=True,
                        help="Path to the claims extracted using VeriScore.")
    parser.add_argument('--knowledge_source_dir', type=str, required=True,
                        help="Path to the directory that contains the knowledge sources for each scientific concept in JSONL format.")
    parser.add_argument('--afv_model', type=str, default="google/gemma-7b-it",
                        help="Name of the Atomic Fact Verification model.")
    parser.add_argument('--data_dir', type=str, default=".cache/factscore",
                        help="Directory to store data files.")
    parser.add_argument('--model_dir', type=str, default=".cache/factscore",
                        help="Directory to store model files.")
    parser.add_argument('--cache_dir', type=str, default=".cache/factscore",
                        help="Directory to store cache files.")
    parser.add_argument('--output_path', type=str, default=".cache/factscore",
                        help="Path to write OFS evaluation results in detail.")
    parser.add_argument('--debug_logger', action='store_true')
    
    args = parser.parse_args()
    ofs_eval = OFSEval(
        extracted_claims_path=args.extracted_claims_path, 
        knowledge_source_dir=args.knowledge_source_dir,
        hf_dataset="rabuahmad/scifactcheck",
        afv_model=args.afv_model,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        debug_logger=args.debug_logger
    )

    # ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # create  filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_path,
        f"ofs_results_{args.afv_model.replace('/', '_')}_{timestamp}.jsonl"
        )

    results_per_model = ofs_eval.run(output_file)

    print(f"Saved OFS evaluation results (JSONL) to {output_file}")


