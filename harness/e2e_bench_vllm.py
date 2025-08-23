import time, sys
try:
    import vllm
    from transformers import AutoTokenizer
except Exception as e:
    print("vLLM / transformers not available. Install inside the container to run this.")
    sys.exit(0)

model = "mistralai/Mistral-7B-Instruct-v0.2"
llm = vllm.LLM(model=model, tensor_parallel_size=1)
tok = AutoTokenizer.from_pretrained(model)

prompts = ["Summarize: " + "lorem ipsum " * 4000] * 4
ts = time.time()
outs = llm.generate(prompts, sampling_params=vllm.SamplingParams(max_tokens=128))
dt = time.time() - ts
print(f"Generated {len(outs)} outputs in {dt:.2f}s")
