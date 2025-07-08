# OpenFActScore

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

OpenFActScore is a modular and extensible framework for evaluating the factual consistency of language model outputs. 
It breaks down generations into atomic facts, retrieves relevant passages, and checks each fact for support using LLMs.
It is built on top of [FActScore](https://github.com/shmsw25/FActScore)


## Install
<!-- ```
conda create -n fs-env python=3.9
conda activate fs-env
pip install -r requirements.txt
``` -->

Make a new Python 3.9+ environment using `virtualenv` or `conda` and install directly from the Github repo

```bash
pip install --upgrade git+https://github.com/lflage/OpenFActScore
python -m spacy download en_core_web_sm
```

For instructions downloading the Wikipedia dump and model productions examples from the original paper, please refer to the original [FActScore](https://github.com/shmsw25/FActScore)  
 
## Running FActScore

We expect running FActScore costs about $1 of the API cost per 100 sentences. For instance, if you have 100 generations, each with 5 sentences on average, it costs $5 in total.

```bash
python -m factscore.factscorer --input_path {input_path} --model_name {estimator_name} --openai_key {openai_key}
```

- `--input_path` can be something like `data/unlabeled/InstructGPT.jsonl`. It should be a `.jsonl` format where each line contains `topic` (a topic entity that corresponds to the Wikipedia title) and `output` (a generation from the model).
- `--afg_model`: HuggingFace name of model selected for Atomic Fact Generation.
- `--afv_model`: HuggingFace name of model selected for Atomic Fact Validation.



**Optional flags**:
- `--openai_key`: File containing OpenAI API Key.
- `--is_npm`: Enables the Non-Parametric model as ensamble
- `--data_dir`: Directory containing knowledge source, etc. `.cache/factscore` by default.
- `--model_dir`: Directory containing Inst-LLAMA weights. Skip if your `model_name` doesn't include `llama`. `.cache/factscore` by default.
- `--cache_dir`: Directory containing cache from API/models. `.cache/factscore` by default.
- `--use_atomic_facts`: If specified, it uses atomic facts saved in the `input_path` file if they exist. This will allow reproducing our results with no (or little if it still uses ChatGPT) cost. You can't specify it if you are running new model generations.
- `--gamma`: A hyperparameter for length penalty. `10` by default. It penalizes the score if the number of facts is less than `gamma`. `10` roughly corresponds to 2 sentences, so would penalize if the generation has less than 2 sentences. Usually, this would not change the ranking between systems unless some systems generate overly short responses all the time (e.g., models trained on NLP datasets without long-form generation tasks may do so). If you would like to turn off the length penalty completely, specify `--gamma 0`.
- `--n_samples`: If specified, it runs the model on a subset of the data.
- `--verbose`: If specified, it shows the progress bar.
- `--print_rate_limit_error`: It specified, it prints out rate limit errors from OpenAI API.
- `--cost_estimate`: This flag decides the type of OpenAI API cost estimation that we provide before calling it. It can be `"consider_cache"` (default) or `"ignore_cache"`.
- `--abstain_detection`: This flag optionally enables automatic detection of abstained responses. By default this is disabled, but it is recommended to add your own function tailored to your model. The currently supported detectors are `"generic"` and `"perplexity_ai"`, and their implementations can be found in [`factscore/abstain_detection.py`](factscore/abstain_detection.py). There are two methods to add your own abstain function: a) clone our GitHub repository to install `factscore` locally (`pip install --editable .`), and then add your function to [`factscore/abstain_detection.py`](factscore/abstain_detection.py) directly; b) process your abstain detection outside our package, and use empty strings in the `output` key for the JSONL file used in `--input_path`.
- `--knowledge_source`: In case the default knowledge source (Wikipedia - 2023/04/01) will not be used, preprocess it using the [instructions below](#To-use-a-custom-knowledge-source), and then specify the knowledge_source name under this flag.


## Input Format

- `topics`: A list of entities or subjects (e.g., Wikipedia page titles).
- `generations`: A list of model-generated responses, one per topic.

Both lists must have the same length.

## To evaluate your own LM

There're two sets of prompt entities, `data/labeled/prompt_entities.txt` (183 entities) and `data/unlabeled/prompt_entities.txt` (500 entities).   
Each line contains the name of the person (which is also a corresponding Wikipedia title).  
You can use the labeled version if you want to be compatible with the data under `data/labeled` (Section 3 and Section 4.2 in the paper), and use the unlabeled version if you want to be compatible with the data under `data/unlabeled` (Section 4.3 in the paper).  

You can prompt your LM with your own prompt (we used `Question: Tell me a bio of <entity>.`) and use the following code.

```python
from factscore.factscorer import FactScorer

fs = FactScorer(
                afg_model="..."
                afv_model="..."
                )

# topics: list of strings (human entities used to generate bios)
# generations: list of strings (model generations)
out = fs.get_score(topics, generations, gamma=10)
print (out["score"]) # FActScore
print (out["init_score"]) # FActScore w/o length penalty
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response
```

Alternatively, you can create a .jsonl file, where each line has `topic` (entity name, exactly same as the one from `.txt` file) and `output` (generation from LM), and then use a command line [above](##Running-FActScore).


Scores for the original FActScore implementation can be seen in columns (A) `AFG=InstructGPT, AFV=chatGPT` and (B) `AFG=InstructGPT, AFV=llama+npm`. 
They have 0.99 Pearson correlation. 
Instructions on reproducing cores for (A) and (B) can be found in [FActScore](https://github.com/shmsw25/FActScore).
We report FActScore obtained by `AFG=allenai/OLMo-2-1124-7B-SFT, AFV=allenai/OLMo-2-1124-7B-SFT` for the following 10 unnanotated data for LLMs prompted in the Biography Writing task.
We report a pearson correlation of over 0.99 between our setting and both (A) and (B)

| Model | % respond | # facts | FActScore from (A) | FActScore from (B) | FActScore from OFS |
|---|---|---|---|---| --- |
| [GPT-4](https://arxiv.org/abs/2303.08774)                                         | 88.2 | 60.8 | 73.1 | 59.9 | 50.1 |
| [ChatGPT](https://openai.com/blog/chatgpt)                                        | 84.2 | 37.0 | 71.6 | 60.4 | 46.5 |
| [Alpaca 65B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                    | 100.0 | 17.1 | 55.6 | 46.3 | 37.1 |
| [InstructGPT](https://openai.com/research/instruction-following)                  | 99.8 | 27.7 | 52.8 | 41.7 | 35.9 |
| [Alpaca 13B](https://crfm.stanford.edu/2023/03/13/alpaca.html)                    | 100.0 | 16.6 | 47.7 | 40.3 | 30.0 |
| [Vicuna 7B](https://lmsys.org/blog/2023-03-30-vicuna/)                            | 91.0 | 45.6 | 38.9 | 36.9 | 29.1|
| [MPT Chat 7B](https://www.mosaicml.com/blog/mpt-7b)                               | 88.8 | 37.3 | 30.1 | 27.9 | 20.7|
| [Oasst Pythia 12B](https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b)   | 100.0 | 39.7 | 25.1 | 20.8 | 16.23 |
| [Dolly 12B](https://huggingface.co/databricks/dolly-v2-12b)                       | 100.0 | 24.6 | 21.7 | 17.1 | 13.5 |
| [StableLM tuned 7B](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)   | 66.6 | 38.0 | 17.3 | 16.3 | 9.2 |

`% respond` (% of responding instead of abstaining from answering) and `# facts` (# of atomic facts per valid response) indicate "factual recall" (how many pieces of information the model gives) and FActScore indicates "factual precision" (how accurate each piece of information the model gives is).

## Using Gemma

To use [Gemma-3-4b](https://huggingface.co/google/gemma-3-4b-it) we recommend disabling TorchDynamo as we encountered unstable behaviour when running inference on multiple atomic facts.

```
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TRANSFORMERS_NO_TORCH_COMPILE=1
```