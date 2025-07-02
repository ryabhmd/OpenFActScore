import sys
import os
import argparse
import json
import pandas as pd
from bert_score import BERTScorer
from pprint import pprint
from tqdm import tqdm

# data_path = "/netscratch/fonseca/HalluEval-RAG/data/atomic_facts/InstructGPT_llama_annotations.jsonl"
# out_file = os.path.basename(data_path).replace(".jsonl","_bertscores.csv")
# out_folder = "../results/metrics/afg_bert_score/"
# out_path = os.path.join(out_folder, out_file)


class AFGBertScores:
    """
    A class to compute and store BERT-based similarity scores between atomic facts 
    and reference sentences from annotated data.
    
    Attributes:
        data_path (str): Path to the input JSONL file containing the annotations.
        out_folder (str): Directory where the output CSV and reports are saved.
        out_file (str): Path to the CSV file storing the computed BERT scores.
        df (pd.DataFrame): DataFrame containing the computed scores.
    """
    def __init__(self, data_path, mode, out_folder="../results/metrics/afg_bert_score/"):
        """
        Initializes the AFGBertScores class, creating necessary directories and 
        loading existing scores or computing new ones.
        
        Args:
            data_path (str): Path to the input JSONL data file.
            out_folder (str, optional): Path to the output folder. Defaults to "../results/metrics/afg_bert_score/".
        """
        self.df = None
        self.compute_map = {
            "original": self.get_scores_original_sentence,
            "human_afs": self.get_score_against_human_afs,
        }
        self.mode = mode
        self.data_path = data_path
        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)

        self.out_file = os.path.join(out_folder, os.path.basename(data_path).replace(".jsonl", "_afg_bert_score.csv"))

    def load_df(self):
        """
        Loads the scores from the output CSV file if it exists; otherwise, computes the scores.
        
        Returns:
            pd.DataFrame: DataFrame containing BERT scores and corresponding annotations.
        """
        if os.path.exists(self.out_file):
            df = pd.read_csv(self.out_file)
            print(df.loc[:, 'f1-score'].mean())
            self.df = df
        else:
            FileNotFoundError(f"File not found: {self.out_file}")

    def compute_scores(self):
        assert self.mode in ['original',"human_afs"], "Mode not found"
        method = self.compute_map.get(self.mode)
        return method()

    def get_scores_original_sentence(self):
        """
        Computes BERT scores for atomic facts against their corresponding reference sentences.
        
        Returns:
            pd.DataFrame: DataFrame containing computed BERT scores.
        """
        scorer = BERTScorer(model_type='bert-base-uncased')

        doc_id, sentences, afs, precision, recall, f1 = [], [], [], [], [], []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, json_f in tqdm(enumerate(f)):
                data_dict = json.loads(json_f)
                for annotation in data_dict['annotations']:
                    ref = annotation['text']
                    for atomic_fact in annotation["model-atomic-facts"]:
                        cur_af = atomic_fact['text']
                        P, R, F = scorer.score([cur_af], [ref])

                        doc_id.append(i)
                        sentences.append(ref)
                        afs.append(cur_af)
                        precision.append(P[0].item())
                        recall.append(R[0].item())
                        f1.append(F[0].item())

        assert len(doc_id) == len(sentences) == len(afs) == len(precision)

        self.df = pd.DataFrame({
            "doc": doc_id,
            "sentence": sentences,
            "atomic_fact": afs,
            "precision": precision,
            "recall": recall,
            "f1-score": f1
        })

        print(self.df.loc[:, 'f1-score'].mean())
        return self.df

    def get_score_against_human_afs(self):
        """ For each model generated atomic fact, get highest BERTSCore between all 
        human generated atomic fact in the same sentence.
        """
        scorer = BERTScorer(model_type='bert-base-uncased')
        doc_id, sentences, list_mg_afs, max_human_af, f1 = [], [], [], [], []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, json_f in tqdm(enumerate(f)):
                data_dict = json.loads(json_f)
                for annotation in data_dict['annotations']:
                    assert "human-atomic-facts" in annotation, "Only possible to use with human-annotated-facts"
                    ref = annotation['text']
                    mg_afs = annotation["model-atomic-facts"]
                    human_afs = annotation["human-atomic-facts"]
                    if not human_afs:
                        continue

                    for mg_af in mg_afs:
                        max_f1 = 0
                        mg_text = mg_af["text"]
                        try:
                            for human_af in human_afs:
                                human_txt = human_af["text"]
                                _, _, F1 = scorer.score([mg_text], [human_txt])
                                if F1 >= max_f1:
                                    max_f1 = F1.item()
                                    max_f1_text = human_txt
                        except TypeError:
                            pprint(annotation)
                            continue

                        doc_id.append(i)
                        sentences.append(ref)
                        list_mg_afs.append(mg_af)
                        max_human_af.append(max_f1_text)
                        f1.append(max_f1)

        self.df = pd.DataFrame({
            "doc": doc_id,
            "sentence": sentences,
            "mg_afs": list_mg_afs,
            "max_human_afs": max_human_af,
            "f1-score": f1
        })

        print(self.df.loc[:, 'f1-score'].mean())
        return self.df

    def get_top5(self):
        """
        Retrieves the top 5 atomic facts with the highest BERT F1 scores.
        
        Returns:
            pd.DataFrame: DataFrame containing the top 5 highest-scoring entries.
        """
        return self.df.nlargest(5, 'f1-score')

    def get_low5(self):
        """
        Retrieves the bottom 5 atomic facts with the lowest BERT F1 scores.
        
        Returns:
            pd.DataFrame: DataFrame containing the 5 lowest-scoring entries.
        """
        return self.df.nsmallest(5, 'f1-score')
    
    def to_csv(self):
        """
        Saves the DataFrame containing BERT scores to a CSV file.
        """
        self.df.to_csv(self.out_file, index=False)

    def get_report(self, report_path=None) -> None:
        """
        Generates a JSON report containing top and bottom 5 scores, mean F1 score, and output CSV file path.
        
        Args:
            report_path (str, optional): Path to save the report. Defaults to replacing ".csv" with "_report.json" in `self.out_file`.
        """
        report_path = report_path if report_path else self.out_file.replace(".csv", "_report.json")

        report_dict = {
            "top5": self.get_top5().to_dict(orient='records'),
            "low5": self.get_low5().to_dict(orient='records'),
            "mean-f1": self.df.loc[:, 'f1-score'].mean(),
            "csv_file": self.out_file,
        }
        print(f"Writing report to: {report_path}")
        with open(report_path, 'w', encoding="utf8") as f:
            json.dump(report_dict, f, indent=4)

if __name__=="__main__":
    os.chdir(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description="Compute BERTScores for Atomic Facts")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input .jsonl file")
    parser.add_argument("--mode", type=str, choices=["original", "human_afs"], required=True,
                        help="Comparison mode: 'original' (against original sentence) or 'human_afs' (against human-annotated AFS)")
    parser.add_argument("--out_folder", type=str, default="../results/metrics/afg_bert_score/",
                        help="Output folder to save CSV and report")

    args = parser.parse_args()

    scorer = AFGBertScores(data_path=args.data_path, mode=args.mode, out_folder=args.out_folder)

    scorer.compute_scores()
    scorer.to_csv()
    scorer.get_report()

