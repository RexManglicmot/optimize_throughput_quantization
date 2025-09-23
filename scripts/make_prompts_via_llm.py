#scripts/make_prompts_via_llmpy
from huggingface_hub import InferenceClient
import os, sys
from app.config import cfg
from tqdm import tqdm
from pathlib import Path

# Obtain attributes from config object
MODEL = cfg.models.primary
COUNT = int(cfg.models.n_prompts)
TOKEN = os.getenv("HF_TOKEN")
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

    client = InferenceClient(provider="hf-inference", token=TOKEN)

    # Use the universal text_generation endpoint (works for chat & non-chat models)
    text = client.text_generation(
        model=MODEL,
        prompt=SYSTEM + "\n\n" + USER,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    

    # COMPLICATED LINE OF CODE HERE????!!!
    lines = [ln.strip().strip("`\"' ") for ln in text.splitlines() if ln.strip()]
    
    # Write to file
    out = Path("data/prompts.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ Wrote {len(lines)} prompts to {out} using {MODEL}")

if __name__ == "__main__":
    main()

# run python3 -m scripts.make_prompts_via_llm