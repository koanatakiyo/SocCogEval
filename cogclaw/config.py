"""Configuration management for CogClaw experiments."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── API Keys ────────────────────────────────────────────────────────────────

def _load_key(filename: str, env_var: str) -> str:
    """Load API key from file in APIkeys/ directory, or from env var."""
    root = Path(__file__).resolve().parent.parent
    key_file = root / "APIkeys" / filename
    if key_file.exists():
        text = key_file.read_text().strip()
        # Strip surrounding quotes and inline comments
        text = text.split("#")[0].strip().strip('"').strip("'")
        return text
    return os.environ.get(env_var, "")


OPENAI_API_KEY = _load_key("api_key_openai.txt", "OPENAI_API_KEY")
DEEPSEEK_API_KEY = _load_key("api_key_deepseek.txt", "DEEPSEEK_API_KEY")
GEMINI_API_KEY = _load_key("api_key_gemini.txt", "GEMINI_API_KEY")
GROK_API_KEY = _load_key("api_key_grok.txt", "GROK_API_KEY")

# ── Local Model Paths ─────────────────────────────────────────────────────
LOCAL_MODEL_CACHE = Path("/mnt/ssd1/yandan/transformer_cache")
QWEN_MODEL_PATH = str(LOCAL_MODEL_CACHE / "qwen3-122b-fp8")
LLAMA_MODEL_PATH = str(LOCAL_MODEL_CACHE / "llama4-scout")

# ── Local vLLM Server ─────────────────────────────────────────────────────
# Local models served via vLLM with OpenAI-compatible endpoint.
# Qwen on port 8000, Llama on port 8001. Override with env vars.
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8001/v1")
LOCAL_API_KEY = "not-needed"  # local vLLM server needs no real key

# ── Frozen API Parameters (from working_plan_v3.md §11.1) ──────────────────

CONVERSATION_TEMPERATURE = 0.7
PROBE_TEMPERATURE = 0.0
CONVERSATION_MAX_TOKENS = 1024
PROBE_MAX_TOKENS = 512
ALLOCATION_MAX_TOKENS = 256

# ── Model Strategy ──────────────────────────────────────────────────────────
# Primary: Qwen 3 122B FP8 (local vLLM) — free unlimited
# Local 2: Llama 4 Scout (local vLLM) — second local model family
# API 1: GPT (OpenAI) — cloud cross-model validation
# API 2: DeepSeek v3 (cloud) — cheap cloud cross-model check
# API 3: Gemini Flash (Google) — fifth model family
# Backup: Grok (xAI) — if needed
PRIMARY_MODEL = "qwen3-122b-fp8"         # Local Qwen 3 122B FP8 (主力)
MODEL_LLAMA = "llama4-scout"             # Local Llama 4 Scout
MODEL_GPT = "gpt-4o-mini"               # GPT (API)
MODEL_DEEPSEEK = "deepseek-chat"         # DeepSeek v3 (API)
MODEL_GEMINI = "gemini-2.0-flash"        # Gemini Flash (API)
MODEL_GROK = "grok-2"                   # Grok (backup)
PILOT_MODEL = PRIMARY_MODEL              # Same as primary (free, unlimited)

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"


# ── Experimental Condition ──────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    """Network topology configuration."""
    topology: str  # "all_to_all", "segregated", "none"
    n_groups: int = 1  # 1 for all_to_all, 2 for segregated
    group_size: int = 4  # agents per group


@dataclass
class SeedConfig:
    """Seed information configuration."""
    use_seed: bool = False
    asymmetric: bool = False
    seed_a: str = ""
    seed_b: str = ""


@dataclass
class InteractionConfig:
    """Interaction configuration."""
    n_rounds: int = 30
    agents_per_conversation: int = 2  # pair conversations
    conversations_per_round: int = 4  # how many pairs per round


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    condition_id: int
    condition_name: str
    scenario: str  # "greenfield" or "riverton"
    n_agents: int = 8
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig("all_to_all"))
    seed: SeedConfig = field(default_factory=SeedConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    model: str = PRIMARY_MODEL
    n_repeats: int = 1
    use_s2: bool = False  # S2 deliberate reasoning
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = d.copy()
        d["network"] = NetworkConfig(**d["network"])
        d["seed"] = SeedConfig(**d["seed"])
        d["interaction"] = InteractionConfig(**d["interaction"])
        return cls(**d)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
