import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    BASE_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    OUTPUT_DIR: str = "./lora_financial_credit_weights"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # QLoRA Parameters
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    TARGET_MODULES: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # Training Parameters
    BATCH_SIZE: int = 2
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 3
    MAX_SEQ_LENGTH: int = 1024

    # Dataset Paths
    TRAIN_DATA_PATH: str = "./data/train_dataset.jsonl"
    EVAL_DATA_PATH: str = "./data/eval_dataset.jsonl"

    # Serving Parameters
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = PipelineSettings()
