# Active Reading Replication
Replication of Meta's [Active Reading](https://arxiv.org/abs/2508.09494) LoRa adapted fine-tuning approach for improving factual recall in small language models. Trains different models on synthetically generated study-strategy data from the facebook/meta-active-reading dataset and evaluates on closed-book factual accuracy on SimpleWikiQA using OpenAI's SimpleQA greading protocol. Our initial tests check whether this holds at 1B scale, but can be scaled further.

## What's here
- `active_reading_training.py` — fine-tunes a base model with LoRA on `facebook/meta-active-reading`
- `active_reading_eval_simple_qa.py` — evaluates closed-book factual recall 
- `notebooks/active_reading.ipynb` — initial exploration notebook with visualizations

## Setup
make a .env file and add HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY
```bash
pip install -r requirements.txt
```

## Training

```bash
python active_reading_training.py --device local
```

## Eval

```bash
python active_reading_eval_simple_qa.py --adapter ./ActiveReading/<run-name>/checkpoint-XXXX --n 200 --output results.json
```

Drop `--adapter` to evaluate the base model as a baseline.

## Results

| Model | is_correct |
|---|---|
| Llama 3.1 8B Base | 7.42 | 
|> repeat | 15.92 |
|> paraphrase | 25.74 |
|> synth QA| 47.87 |
|> Active Reading (task-agnostic) | 63.33 |
|> Active Reading (task-specific) | 66.25 |
|> paraphrase + synth QA + AR | 66.66 |
| **Llama 3.2 1B Base | 1.0 |
| **Llama 3.2 1B AR (13105 train samples) | 2.0 |

_** only tested on 200 questions_
## Reference

> Shi et al., *Active Reading: Strategies for Robust Knowledge Retention in Language Models*, Meta 2025.
