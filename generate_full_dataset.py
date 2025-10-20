#!/usr/bin/env python3
"""
Complete data generation pipeline for DPO training.
This script generates math problems, creates negative responses using NanoGPT,
and combines them with programmatic positive responses.
"""

import sys
import os
sys.path.append(os.path.abspath("."))
import torch
import pickle
import json
from model import GPT, GPTConfig
from tqdm import tqdm
import argparse

def generate_problems(count=10000):
    """Generate math problems programmatically"""
    import random

    problems = []

    # Generate arithmetic problems (half)
    for _ in range(count // 2):
        op = random.choice(['+', '-', '*', '/'])

        if op == '+':
            a, b = random.randint(1, 99), random.randint(1, 99)
            answer = a + b
            problem = f"{a}+{b}=?"
            explanation = f"The answer is {answer} because {a}+{b} equals {answer}."
        elif op == '-':
            a, b = random.randint(10, 99), random.randint(1, 99)
            if a < b: a, b = b, a
            answer = a - b
            problem = f"{a}-{b}=?"
            explanation = f"The answer is {answer} because {a}-{b} equals {answer}."
        elif op == '*':
            a, b = random.randint(1, 15), random.randint(1, 15)
            answer = a * b
            problem = f"{a}*{b}=?"
            explanation = f"The answer is {answer} because {a}*{b} equals {answer}."
        elif op == '/':
            b = random.randint(2, 12)
            answer = random.randint(2, 20)
            a = b * answer
            problem = f"{a}/{b}=?"
            explanation = f"The answer is {answer} because {a}/{b} equals {answer}."

        problems.append((problem, explanation))

    # Generate algebra problems (half)
    for _ in range(count - len(problems)):
        problem_type = random.choice(['add', 'mult', 'sub'])

        if problem_type == 'add':
            x = random.randint(1, 50)
            a = random.randint(1, 50)
            b = x + a
            problem = f"x+{a}={b},x=?"
            explanation = f"The answer is {x} because {b}-{a} equals {x}."
        elif problem_type == 'mult':
            x = random.randint(1, 20)
            a = random.randint(2, 12)
            b = a * x
            problem = f"{a}*x={b},x=?"
            explanation = f"The answer is {x} because {b}/{a} equals {x}."
        elif problem_type == 'sub':
            x = random.randint(1, 30)
            b = random.randint(1, 30)
            a = x + b
            problem = f"{a}-x={b},x=?"
            explanation = f"The answer is {x} because {a}-{b} equals {x}."

        problems.append((problem, explanation))

    random.shuffle(problems)
    return problems

def load_model():
    """Load the pretrained NanoGPT model"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load("sft/gpt.pt", map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    gpt = GPT(gptconf)

    # Clean up state dict
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    gpt.load_state_dict(state_dict)
    gpt.to(device).eval()

    # Load tokenizer
    with open("sft/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s):
        return [stoi.get(c, 0) for c in s]

    def decode(l):
        return ''.join([itos.get(i, '') for i in l])

    return gpt, encode, decode, device

def generate_negative_response(gpt, encode, decode, device, problem, max_new_tokens=30):
    """Generate negative response for a math problem"""
    try:
        problem_ids = encode(problem)
        x = torch.tensor(problem_ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            y, _ = gpt.generate(x, max_new_tokens, temperature=0.8, top_k=200)

        response = decode(y[0].tolist())
        generated_part = response[len(problem):].strip()
        return f"{problem} {generated_part}"

    except Exception as e:
        return f"{problem} Sorry, I do not know!"

def main():
    parser = argparse.ArgumentParser(description='Generate DPO training dataset')
    parser.add_argument('--count', type=int, default=10000, help='Number of problems to generate')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')
    args = parser.parse_args()

    print(f"Generating {args.count} math problems...")
    problems = generate_problems(args.count)

    print("Loading pretrained model...")
    gpt, encode, decode, device = load_model()

    print(f"Generating negative responses...")
    pos_neg_pairs = []

    for i, (problem, positive_answer) in enumerate(tqdm(problems)):
        # Generate negative response
        negative_response = generate_negative_response(gpt, encode, decode, device, problem)

        # Create positive response
        positive_response = f"{problem} {positive_answer}"

        # Add to dataset
        pos_neg_pairs.append({
            "negative": negative_response,
            "positive": positive_response
        })

    # Save the complete dataset
    output_file = 'dpo/pos_neg_pairs.json'
    with open(output_file, 'w') as f:
        json.dump(pos_neg_pairs, f, indent=2)

    print(f"\nGenerated {len(pos_neg_pairs)} positive-negative pairs")
    print(f"Saved to {output_file}")
    print("\nSample pairs:")
    for i in range(3):
        print(f"\nPair {i+1}:")
        print(f"  Negative: {pos_neg_pairs[i]['negative']}")
        print(f"  Positive: {pos_neg_pairs[i]['positive']}")

if __name__ == "__main__":
    main()