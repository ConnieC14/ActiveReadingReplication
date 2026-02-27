"""
SimpleQA Evaluation Script: Measuring short-form factuality in large language models 
Authors: Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus
https://cdn.openai.com/papers/simpleqa.pdf

Evaluates a fine-tuned (or base) Llama adapter against the SimpleWikiQA subset
using the exact grading protocol from the OpenAI simple-evals repo:
  https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py

Grading
  - Model answers each SimpleWikiQA question in closed-book
  - GPT-4.1 judges each answer as CORRECT / INCORRECT / NOT_ATTEMPTED
  - Final score = % CORRECT  (paper target: 66.25% for Active Reading task-specific)

Usage
  python active_reading_eval_simple_qa.py --adapter ./active-reading-llama-1b --n 200 --output eval_results.json
  # Base model only (no adapter), useful as a baseline:
  python active_reading_eval_simple_qa.py --n 200 --output baseline_results.json
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
import random
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

parser = argparse.ArgumentParser(description="Evaluate a model on SimpleWikiQA")
parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to LoRA adapter directory (if not provided, evaluates base model only)"
    )
parser.add_argument(
    "--grader_model",
    type=str,
    default="gpt-4.1",
    help="Model to user for grading (default: gpt-4.1, same as paper)",
)
parser.add_argument(
    "--base_model",
    type=str,
    default="meta-llama/Llama-3.2-1B",
    help="Base model for fine-tuning (default: meta-llama/Llama-3.2-1B)",
)
parser.add_argument(
        "--n", type=int, default=200,
        help="Number of questions to evaluate (default: all ~3449)"
)
parser.add_argument(
        "--output", type=str, default=f"./eval_results_{datetime.now().strftime('%Y%m%d-%H%M')}.json",
        help="Path to save JSON results"
)

args = parser.parse_args()

BASE_MODEL       = args.base_model
HF_TOKEN         = os.getenv("HF_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
GRADER_MODEL     = args.grader_model         
SIMPLEQA_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"

# Generation config
MAX_NEW_TOKENS  = 64
QUESTION_PROMPT = "{question}"
GRADER_TEMPLATE = """
    Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
    First, I will give examples of each grade, and then you will grade a new example.


    The following are examples of CORRECT predicted answers.
    ```
    Question: What are the names of Barack Obama's children?
    Gold target: Malia Obama and Sasha Obama
    Predicted answer 1: sasha and malia obama
    Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
    Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
    ```
    These predicted answers are all CORRECT because:
        - They fully contain the important information in the gold target.
        - They do not contain any information that contradicts the gold target.
        - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
        - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


    The following are examples of INCORRECT predicted answers.
    ```
    Question: What are the names of Barack Obama's children?
    Gold target: Malia and Sasha
    Predicted answer 1: Malia.
    Predicted answer 2: Malia, Sasha, and Susan.
    Predicted answer 3: Barack Obama does not have any children.
    Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
    Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
    Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
    Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
    ```
    These predicted answers are all INCORRECT because:
        - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


    The following are examples of NOT_ATTEMPTED predicted answers.
    ```
    Question: What are the names of Barack Obama's children?
    Gold target: Malia and Sasha
    Predicted answer 1: I don't know.
    Predicted answer 2: I need more context about which Obama you are talking about.
    Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
    Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
    ```
    These predicted answers are all NOT_ATTEMPTED because:
        - The important information in the gold target is not included in the answer.
        - No statements in the answer contradict the gold target.


    Also note the following things:
    - For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
        - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
        - Predicted answers "100k" and "113k" are INCORRECT. 
        - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
    - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
        - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
    - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
        - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
        - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
        - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
        - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
    - Do not punish for typos in people's name if it's clearly the same name. 
        - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


    Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
    ```
    Question: {question}
    Gold target: {target}
    Predicted answer: {predicted_answer}
    ```

    Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    C: NOT_ATTEMPTED

    Just return the letters "A", "B", or "C", with no text around it.
""".strip()

GRADE_LETTER_TO_STRING = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
STRING_TO_GRADE_LETTER = {v: k for k, v in GRADE_LETTER_TO_STRING.items()}


def load_simplewikiqa(csv_path: str, n: int):

    df = pd.read_csv(csv_path)
    print(f"Total SimpleQA questions: {len(df)}")

    wiki_rows = [row.to_dict() for _, row in df.iterrows()]
    if n:
        rng = random.Random(0)
        wiki_rows = rng.sample(wiki_rows, min(n, len(wiki_rows)))
        print(f"Subsampled to {len(wiki_rows)} questions")

    return wiki_rows

def grade_answer(client: OpenAI, question: str, target: str, predicted: str,
                 retries: int = 3):
    prompt = GRADER_TEMPLATE.format(
        question=question,
        target=target,
        predicted_answer=predicted,
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GRADER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2048, # https://github.com/openai/simple-evals/blob/main/sampler/responses_sampler.py
            )
            grading_response  = response.choices[0].message.content.strip().upper()
            letter = re.search(r"(A|B|C)", grading_response )
            if letter:
                return letter.group(0)
            # Sometimes model returns the full word instead of a letter
            for grade in ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]:
                if grade in grading_response:
                    return STRING_TO_GRADE_LETTER[grade]
            print(f"  Unexpected grader response: '{grading_response}'")
            return "B"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Grader failed after {retries} attempts: {e}")
                return "C"


def load_model(adapter_path, base_model=BASE_MODEL):
    """
    Load base model with optional LoRA adapter.
    If adapter_path is None, evaluates the base model (baseline).
    """
    print(f"\nLoading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
    )

    if adapter_path:
        adapter_path = Path(adapter_path).resolve()
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("  Adapter merged into base model")
    else:
        print("  No adapter: evaluating base model")

    model.eval()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer


def generate_answer(model, tokenizer, question: str):
    prompt = QUESTION_PROMPT.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Truncate at newline — we only want the first line
    answer = answer.split("\n")[0].strip()
    return answer


def run_eval(adapter_path, n, output_path: str):
    # Load data
    questions = load_simplewikiqa(SIMPLEQA_CSV_URL, n=n)

    # Load model
    model, tokenizer = load_model(adapter_path)

    # OpenAI client for grading
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    results = []
    counts  = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}

    print(f"\nEvaluating {len(questions)} questions ...")
    for i, row in enumerate(tqdm(questions, desc="Evaluating")):
        question = row["problem"]
        target   = row["answer"]

        predicted = generate_answer(model, tokenizer, question)

        grade = GRADE_LETTER_TO_STRING[grade_answer(openai_client, question, target, predicted)]
        counts[grade] += 1

        results.append({
            "question":  question,
            "target":    target,
            "predicted": predicted,
            "grade":     grade,
        })

        # Print running score every 100 questions
        if (i + 1) % 100 == 0:
            done = i + 1
            pct_correct = counts["CORRECT"] / done * 100
            print(f"  [{done}/{len(questions)}] "
                  f"CORRECT={counts['CORRECT']} ({pct_correct:.1f}%) "
                  f"INCORRECT={counts['INCORRECT']} "
                  f"NOT_ATTEMPTED={counts['NOT_ATTEMPTED']}")

    # Final scores
    total = len(results)
    pct_correct      = counts["CORRECT"]      / total * 100
    pct_incorrect    = counts["INCORRECT"]    / total * 100
    pct_not_attempted = counts["NOT_ATTEMPTED"] / total * 100

    # After counting, add these to your final score block
    attempted = counts["CORRECT"] + counts["INCORRECT"]
    accuracy_given_attempted = counts["CORRECT"] / attempted if attempted > 0 else 0
    f1 = (
        2 * accuracy_given_attempted * pct_correct/100
        / (accuracy_given_attempted + pct_correct/100)
        if (accuracy_given_attempted + pct_correct/100) > 0
        else 0
    )

    print()
    print(f"  SimpleWikiQA Evaluation Results")
    print(f"  Model    : {adapter_path or BASE_MODEL + ' (base)'}")
    print(f"  Questions: {total}")
    print()
    print(f"  CORRECT      : {counts['CORRECT']:4d}  ({pct_correct:.1f}%)")
    print(f"  INCORRECT    : {counts['INCORRECT']:4d}  ({pct_incorrect:.1f}%)")
    print(f"  NOT_ATTEMPTED: {counts['NOT_ATTEMPTED']:4d}  ({pct_not_attempted:.1f}%)")
    print(f"  accuracy_given_attempted: {accuracy_given_attempted:.3f}")
    print(f"  F1                      : {f1:.3f}")
    print()
    print(f"\n  Paper target Active Reading task-specific: 66.25%")
    print(f"  Paper baseline Llama 3.1 8B Base:          7.42%")

    # Save results
    output = {
        "model":          adapter_path or BASE_MODEL,
        "base_model":     BASE_MODEL,
        "grader_model":   GRADER_MODEL,
        "n_questions":    total,
        "counts":         counts,
        "pct_correct":    round(pct_correct, 2),
        "pct_incorrect":  round(pct_incorrect, 2),
        "pct_not_attempted": round(pct_not_attempted, 2),
        "accuracy_given_attempted": round(accuracy_given_attempted, 3),
        "f1":             round(f1, 3),
        "completed":      datetime.now().isoformat(),
        "results":        results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n Full results saved to {output_path}")


if __name__ == "__main__":
    print("\nStarting SimpleQA evaluation")
    run_eval(
        adapter_path=args.adapter,
        n=args.n,
        output_path=args.output,
    )