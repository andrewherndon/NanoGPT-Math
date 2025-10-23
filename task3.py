import re, json, math, random, time
from typing import List, Dict

policy.eval()

def generate(model, prompt: str, max_new_tokens: int = 64) -> str:
    x = encode(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx), dim=1)
    return decode(x[0])

def solve_ground_truth(prompt: str):
    try:
        m = re.fullmatch(r"\s*(-?\d+)\s*\+\s*(-?\d+)\s*=\?\s*", prompt)
        if m: return int(m.group(1)) + int(m.group(2))
        m = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*=\?\s*", prompt)
        if m: return int(m.group(1)) - int(m.group(2))
        m = re.fullmatch(r"\s*(-?\d+)\s*\*\s*(-?\d+)\s*=\?\s*", prompt)
        if m: return int(m.group(1)) * int(m.group(2))
        m = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*=\?\s*", prompt)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b == 0: return None
            return a // b
        m = re.fullmatch(r"\s*(-?\d+)\s*\*\s*x\s*=\s*(-?\d+)\s*,\s*x\s*=\?\s*", prompt)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a == 0: return None
            return b // a
        m = re.fullmatch(r"\s*(-?\d+)\s*\*\s*x\s*\+\s*(-?\d+)\s*=\s*(-?\d+)\s*,\s*x\s*=\?\s*", prompt)
        if m:
            a, c, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a == 0: return None
            return (d - c) // a
        m = re.fullmatch(r"\s*x\s*/\s*(-?\d+)\s*=\s*(-?\d+)\s*,\s*x\s*=\?\s*", prompt)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return a * b
        m = re.fullmatch(r"\s*\(\s*x\s*\+\s*(-?\d+)\s*\)\s*\*\s*(-?\d+)\s*=\s*(-?\d+)\s*,\s*x\s*=\?\s*", prompt)
        if m:
            m_val, a, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a == 0: return None
            return (c // a) - m_val
    except Exception:
        return None
    return None

def parse_numeric_answer(text: str):
    nums = re.findall(r"[-+]?\d+", text)
    if not nums: return None
    return int(nums[-1])

def evaluate_on_tests(model, tests: List[str], max_new_tokens: int = 72) -> List[Dict]:
    results = []
    for t in tests:
        raw = generate(model, t + " ", max_new_tokens=max_new_tokens).strip()
        pred = parse_numeric_answer(raw)
        truth = solve_ground_truth(t)
        results.append({
            "prompt": t,
            "output": raw,
            "parsed_pred": pred,
            "truth": truth,
            "correct": pred == truth,
        })
    return results

tests = [
    "12+47=?", "91-58=?", "9*8=?", "72/9=?",
    "7*x+5=54, x=?", "3*x= -24, x=?", "x/6= -7, x=?",
    "(x+3)*4=20, x=?", "15+27=?", "100-1=?"
]

def make_random_suite(k=20):
    s = []
    for _ in range(k):
        t = random.choice(["add","sub","mul","div","ax_b_eq_c","ax_eq_b","x_div_a_eq_b","two_step"])
        if t == "add":
            a,b = random.randint(0,999), random.randint(0,999); s.append(f"{a}+{b}=?")
        elif t == "sub":
            a,b = random.randint(0,999), random.randint(0,999); s.append(f"{a}-{b}=?")
        elif t == "mul":
            a,b = random.randint(0,99), random.randint(0,99); s.append(f"{a}*{b}=?")
        elif t == "div":
            b = random.randint(1,99); ans = random.randint(1,99); a=b*ans; s.append(f"{a}/{b}=?")
        elif t == "ax_b_eq_c":
            a = random.randint(1,12); x = random.randint(-50,50); c_add = random.randint(-20,20); c = a*x + c_add
            s.append(f"{a}*x+{c_add}={c}, x=?")
        elif t == "ax_eq_b":
            a = random.randint(1,12); x = random.randint(-50,50); b = a*x; s.append(f"{a}*x={b}, x=?")
        elif t == "x_div_a_eq_b":
            a = random.randint(1,12); b = random.randint(-50,50); s.append(f"x/{a}={b}, x=?")
        else:
            a = random.randint(1,12); m = random.randint(-20,20); x = random.randint(-50,50); c = (x+m)*a
            s.append(f"(x+{m})*{a}={c}, x=?")
    return s

tests += make_random_suite(30)

start = time.time()
results = evaluate_on_tests(policy, tests)
dur = time.time() - start
correct = sum(r["correct"] for r in results)
acc = 100.0 * correct / len(results)

print(f"Evaluated {len(results)} prompts in {dur:.2f}s on {device}. Accuracy: {acc:.2f}%")
print("-"*60)
for r in results[:20]:
    print(f"Prompt: {r['prompt']}")
    print(f"Output: {r['output']}")
    print(f"Parsed: {r['parsed_pred']} | Truth: {r['truth']} | Correct: {r['correct']}")
    print("-"*60)

eval_path = os.path.join(DPO_DIR, "eval_results.json")
with open(eval_path, "w") as f:
    json.dump({
        "device": device,
        "accuracy": acc,
        "correct": correct,
        "total": len(results),
        "results": results
    }, f, indent=2)
print("Saved results to:", eval_path)