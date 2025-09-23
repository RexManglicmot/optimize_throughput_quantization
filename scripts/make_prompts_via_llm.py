#scripts/make_prompts_via_llmpy
from huggingface_hub import InferenceClient
import os, sys
from app.config import cfg
from tqdm import tqdm

# Obtain attributes from config object
MODEL = str(cfg.models.primary)
COUNT = int(cfg.models.n_prompts)
TOKEN = cfg.HF_TOKEN
MAX_TOKENS = int(cfg.llm_script.max_tokens)
TEMPERATURE = float(cfg.llm_script.temperature)
TOP_P = float(cfg.llm_script.top_p)

# Set params for LLM to use
SYSTEM = "You are a data generator. Output ONLY raw prompts, one per line. No numbering, no bullets, no quotes."
USER = f"""Generate {COUNT} diverse prompts (1–2 sentences each) for throughput testing.
Mix styles: explain, summarize, compare, steps/how-to, pros/cons, classify, reasoning, FAQ.
Keep it PG/safe. Output EXACTLY one prompt per line. No extra text."""

# Invoke script
def main():
    
    # Double check if token is loaded, if not, make print statement and exit
    if not TOKEN:
        sys.exit("Set HF_TOKEN in environment.")

    client = InferenceClient(model=MODEL, token=TOKEN)

    # Try chat; 
    # Put tqdm to measure progress....dont know if it is right?

    resp = tqdm(client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":USER}],
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P,
    ))

    text = resp.choices[0].message.content
    

    # COMPLICATED LINE OF CODE HERE????!!!
    lines = [ln.strip().strip("`\"' ") for ln in text.splitlines() if ln.strip()]
    
    # Write to file
    os.makedirs("data", exist_ok=True)
    with open("data/prompts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f" Wrote {len(lines)} prompts to data/prompts.txt using {MODEL}")

if __name__ == "__main__":
    main()

# run python3 -m scripts.make_prompts_via_llm