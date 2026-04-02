"""YAML-driven experiment configuration for AEL framework."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolConfig:
    """Configuration for tool selection."""
    enabled_tools: list[str] = field(default_factory=list)  # empty = all
    selection_method: str = "thompson"  # "thompson", "uniform", "fixed"
    max_tools_per_step: int = 5
    evolve: bool = True
    per_tool_selection: bool = False  # EAEL v2: per-tool Thompson instead of preset-based


@dataclass
class PlannerConfig:
    """Configuration for planner selection."""
    enabled_planners: list[str] = field(default_factory=lambda: [
        "sequential", "decompose", "adaptive"
    ])
    selection_method: str = "linucb"  # "linucb", "uniform", "fixed"
    fixed_planner: str = "sequential"  # used when selection_method="fixed"
    evolve: bool = True
    linucb_alpha: float = 1.0


@dataclass
class MemoryConfig:
    """Configuration for memory system."""
    enabled_tiers: list[str] = field(default_factory=lambda: [
        "episodic", "semantic", "procedural"
    ])
    retrieval_top_k: int = 5
    retrieval_method: str = "embedding"  # "embedding", "keyword", "hybrid"
    write_quality_threshold: float = 0.3
    eviction_min_score: float = 0.2
    max_entries_per_tier: int = 500
    evolve: bool = True
    usefulness_mode: str = "raw_sign"  # "raw_sign" (outcome>0) or "differential" (outcome>rolling_avg)
    backend: str = "local"  # "local" or "xmem"
    xmem_url: str = "http://127.0.0.1:8200"
    xmem_agent_id: str = "ael"


@dataclass
class CreditConfig:
    """Configuration for credit assignment."""
    method: str = "fcc"  # "fcc", "structural", "uniform", "llm", "llm_fcc", "llm_uniform", "curriculum", "curriculum_llm"
    use_counterfactual: bool = True
    use_shapley: bool = True
    shapley_interval: int = 20  # episodes between full Shapley computations
    baseline_planner: str = "sequential"
    baseline_tools: list[str] = field(default_factory=list)
    baseline_memory_policy: str = "none"
    # Curriculum credit: start uniform, switch to FCC after N episodes
    curriculum_switch_episode: int = 60
    # FCC weight tuning
    fcc_weight_structural: float = 0.2
    fcc_weight_counterfactual: float = 0.3
    fcc_weight_shapley: float = 0.5
    # LLM-uniform blend: alpha=1.0 is pure uniform, alpha=0.0 is pure LLM
    llm_blend_alpha: float = 0.5


@dataclass
class ReflectionConfig:
    """Configuration for daily reflection and planner evolution."""
    enabled: bool = True
    llm_model: str = ""  # defaults to experiment's llm_model
    meta_llm_model: str = ""  # stronger model for reflection/evolution (e.g., Sonnet 4.5)
    structural_problem_threshold: int = 3  # consecutive day failures before flagging
    max_planner_pool_size: int = 10
    planner_probation_days: int = 5  # days before pruning a new planner
    evolve_planners: bool = True  # allow LLM code generation of new planners
    evolution_sessions_per_day: int = 2  # 1=daily, 2=AM/PM, 4=per-bar


@dataclass
class ColdStartConfig:
    """Configuration for LLM cold-start initialization."""
    enabled: bool = True
    llm_model: str = ""  # defaults to experiment's llm_model


@dataclass
class SkillConfig:
    """Configuration for the skills learning system."""
    enabled: bool = True
    extract_tools: bool = True
    extract_templates: bool = True
    extract_strategies: bool = True
    extraction_interval_days: int = 3       # extract strategies every N days
    min_procedural_confidence: float = 0.6  # min success rate before extracting
    max_skills_per_type: int = 15
    skill_probation_episodes: int = 20


@dataclass
class BenchmarkConfig:
    """Configuration for the finance benchmark."""
    tickers: list[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    train_end_date: str = ""   # end of train phase (exclusive); val starts next day
    val_end_date: str = ""     # end of val phase (exclusive); test starts next day
    prediction_horizons: list[int] = field(default_factory=lambda: [1, 7, 14])
    usstock_path: str = ""
    frequency: str = "1d"       # "1d" or "1h"
    bars_per_day: int = 1       # 1 for daily, 4 for hourly (9:30, 11:30, 13:30, 15:30)
    flat_threshold: float = 0.005  # 0.5% for daily, 0.1% for hourly


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    name: str = "default"
    seed: int = 42
    num_episodes: int = 100
    log_dir: str = "logs"

    tools: ToolConfig = field(default_factory=ToolConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    cold_start: ColdStartConfig = field(default_factory=ColdStartConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)

    # Warm-up: during first N episodes, behave as tool_only (no memory/reflection/evolution)
    warm_up_episodes: int = 0  # 0 = disabled

    # LLM settings
    llm_model: str = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        cfg = cls()
        for key, val in d.items():
            if key == "tools" and isinstance(val, dict):
                cfg.tools = ToolConfig(**val)
            elif key == "planner" and isinstance(val, dict):
                cfg.planner = PlannerConfig(**val)
            elif key == "memory" and isinstance(val, dict):
                cfg.memory = MemoryConfig(**val)
            elif key == "credit" and isinstance(val, dict):
                cfg.credit = CreditConfig(**val)
            elif key == "benchmark" and isinstance(val, dict):
                cfg.benchmark = BenchmarkConfig(**val)
            elif key == "reflection" and isinstance(val, dict):
                cfg.reflection = ReflectionConfig(**val)
            elif key == "cold_start" and isinstance(val, dict):
                cfg.cold_start = ColdStartConfig(**val)
            elif key == "skills" and isinstance(val, dict):
                cfg.skills = SkillConfig(**val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    def to_dict(self) -> dict:
        """Serialize config to dict (for logging)."""
        from dataclasses import asdict
        return asdict(self)
