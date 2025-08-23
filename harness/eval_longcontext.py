import random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def make_prompt(N=16000, needle="SECRET42"):
    chunks = ["blah"] * N
    pos = random.randrange(N)
    chunks[pos] = needle
    return " ".join(chunks), pos

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    text, pos = make_prompt()
    prompt = text + "\nWhere is the secret token?"
    out = model.generate(**tok(prompt, return_tensors="pt").to(device), max_new_tokens=32)
    print(tok.decode(out[0])[:400])
