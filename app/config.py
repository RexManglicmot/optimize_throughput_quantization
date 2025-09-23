import os
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Path to config.yaml (root of project)
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class DotDict(dict):
    """dict → object with attribute access"""
    def __getattr__(self, k):
        v = self.get(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: Path = CONFIG_PATH):
    """Load YAML config into a dictionary and normalize paths."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # resolve device
    if cfg["models"].get("device", "auto") == "auto":
        cfg["models"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    root = path.parent
    # normalize project paths
    if "project" in cfg:
        if "input_dir" in cfg["project"]:
            cfg["project"]["input_dir"] = str((root / cfg["project"]["input_dir"]).resolve())
        if "output_dir" in cfg["project"]:
            cfg["project"]["output_dir"] = str((root / cfg["project"]["output_dir"]).resolve())

    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            cfg["paths"][k] = str((root / v).resolve())
    
    return DotDict(cfg)

# Global config object
cfg = load_config()

# Hugging Face API key (optional if using private models)
HF_TOKEN = os.getenv("HF_TOKEN")

# Debug mode
if __name__ == "__main__":
    print("Config loaded successfully")
    print("Project:", cfg.project.name)
    print("Input dir:", cfg.project.input_dir)
    print("Output dir:", cfg.project.output_dir)
    print("Primary model:", cfg.models.primary)
    print("Device:", cfg.models.device)
    print("hf_token found:", cfg.hf_token is not None)
    print("Precisions to sweep", cfg.models.presions)
    print("Batches to test are", cfg.models.batch_sizes)
    print("For synthetic data, max tokens are ", cfg.llm_script.max_tokens, "and the temp is",
          cfg.llm_script.temperature)

# It works
# 9/22