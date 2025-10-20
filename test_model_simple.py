#!/usr/bin/env python3
"""
Simple test for the fine-tuned DPO model (no interactive mode)
"""

import sys
import os
sys.path.append(os.path.abspath("."))
import torch
import pickle
from model import GPT, GPTConfig

def load_finetuned_model():
    """Load the DPO fine-tuned model"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Load the fine-tuned model
    ckpt_path = "dpo/dpo.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    gpt = GPT(gptconf)

    # Load state dict
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint['model_state_dict']

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

def test_problem(gpt, encode, decode, device, problem, max_new_tokens=50):
    """Test a single math problem"""
    try:
        problem_ids = encode(problem)
        x = torch.tensor(problem_ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            y, _ = gpt.generate(x, max_new_tokens, temperature=0.8, top_k=200)

        response = decode(y[0].tolist())
        generated_part = response[len(problem):].strip()
        return generated_part

    except Exception as e:
        return f"Error: {e}"

def main():
    print("Loading fine-tuned DPO model...")
    gpt, encode, decode, device = load_finetuned_model()

    # Test problems from the assignment
    test_problems = [
        "17+19=?",
        "3*17=?",
        "72/4=?",
        "72-x=34,x=?",
        "x*11=44,x=?",
        "5+8=?",
        "20-6=?",
        "x+3=10,x=?",
        "4*5=?",
        "15/3=?"
    ]

    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL ON MATH PROBLEMS")
    print("="*60)

    correct = 0
    total = len(test_problems)

    # Expected answers for verification
    expected = ["36", "51", "18", "38", "4", "13", "14", "7", "20", "5"]

    for i, problem in enumerate(test_problems):
        print(f"\nProblem {i+1}: {problem}")
        response = test_problem(gpt, encode, decode, device, problem)
        print(f"Response: {response}")

        # Extract just the number from response for comparison
        import re
        numbers = re.findall(r'\d+', response)
        if numbers and numbers[0] == expected[i]:
            print("✅ CORRECT")
            correct += 1
        else:
            print(f"❌ WRONG (expected: {expected[i]})")

        print("-" * 40)

    print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    print("Testing completed!")

if __name__ == "__main__":
    main()