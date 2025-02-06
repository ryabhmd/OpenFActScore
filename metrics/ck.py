import json
import os
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from envs import ROOT

PATH = "/home/lucas_dfki/Documentos/repos/HalluEval-RAG/data/afv_json/InstructGPT_factscore_output_human-atomic-facts.json"
rel_path = 'data/labeled/InstructGPT.jsonl'


class CKScore():
    def __init__(self,
                path,
                human_path = os.path.join(ROOT, 'data/labeled/InstructGPT.jsonl')
    ):
        self.path = path
        self.human_path = human_path
        self.human_annotations = self.get_human_annotations()
        self.machine_annotations = self.get_machine_annotations()

    def replace(self, a):
        """
        replaces original labeling with True for "S" (supported) or False for IR (irrelevant) or NS(not-supported)
        """
        if a == "IR":
            return False
        if a == "S":
            return True
        if a == "NS":
            return False

    def get_human_annotations(self):
        """
        Replacing original FactScore manual labels with machine labels 
        i.e. IR, NS, S for True or False by passing all atomic facts and decisions to a
        list of {"text": str, "label": bool}
        """
        human_evals = []
        with open(self.human_path , 'r', encoding='utf-8') as f:
            for line in f:
                i = json.loads(line)
                if i['annotations'] is None:
                    # print(i["topic"])
                    continue
                for a in i['annotations']:
                    if not a['is-relevant']:
                        continue
                    for fact in a["human-atomic-facts"]:
                        human_evals.append({"text": fact["text"],
                                            "label": self.replace(fact.get("label")) })
        return human_evals

    def get_machine_annotations(self):
        # Load Machine annotated data
        # Pass model decisionas as list of {"atom": str, "is_supported": bool"}
        model_evals = []
        with open(self.path, 'r', encoding='utf-8') as jsl:
            model_evals = json.load(jsl)
                
        models = [atom for doc in model_evals["decisions"] for atom in doc] 
        return models

    def get_cks(self):
        assert len(self.machine_annotations) == len(self.human_annotations)
        for machine, human in zip(self.machine_annotations, self.human_annotations):
            assert machine["atom"] == human["text"]

        y1 = [i["is_supported"] for i in self.machine_annotations]
        y2 = [i["label"] for i in self.human_annotations]

        # TODO Values in list are bool
        assert all(isinstance(item, bool) for item in y1)
        assert all(isinstance(item, bool) for item in y2)

        ck_score = cohen_kappa_score(y1, y2)
        return ck_score
    
    # TODO: Report

if __name__ == "__main__":
    # Path to the machine annotations JSON file
    path_to_machine_annotations = "../data/labeled/InstructGPT_factscore_output_human-atomic-facts copy.json"

    # Initialize CKScore object
    ck_score_calculator = CKScore(path_to_machine_annotations)

    # Calculate Cohen's Kappa Score
    print("Calculating Cohen's Kappa Score...")
    score = ck_score_calculator.get_cks()
    print(f"Cohen's Kappa Score: {score}")
