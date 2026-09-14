import os
from typing import Any, Dict, Tuple
from datasets import Dataset, load_dataset
import pandas as pd
from src.config import settings


class FinancialDatasetPreparer:
    """Handles data ingestion, preprocessing, and ChatML formatting for Financial Credit-Risk SFT."""

    SYSTEM_PROMPT: str = (
        "You are a Senior Financial Risk Analyst. Analyze corporate disclosures, "
        "determine credit risk impact (POSITIVE, NEGATIVE, NEUTRAL), and extract financial metrics."
    )

    @classmethod
    def format_instruction_prompt(cls, sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Formats sample data into standard ChatML / Instruction format for Supervised Fine-Tuning.
        """
        user_msg = (
            f"Corporate Disclosure:\n{sample['context']}\n\n"
            f"Task: Assess liquidity risk and state impact."
        )
        assistant_msg = sample["ground_truth_answer"]

        formatted_text = (
            f"<|im_start|>system\n{cls.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
        return {"text": formatted_text}

    @classmethod
    def load_and_preprocess(
        cls,
        train_path: str = settings.TRAIN_DATA_PATH,
        eval_path: str = settings.EVAL_DATA_PATH,
    ) -> Tuple[Dataset, Dataset]:
        """
        Loads training and validation datasets from disk if present;
        otherwise falls back to synthetic curated financial disclosure samples.
        """
        # Load from files if both exist
        if os.path.exists(train_path) and os.path.exists(eval_path):
            print(f"[+] Loading datasets from {train_path} and {eval_path}")
            data_files = {"train": train_path, "test": eval_path}
            raw_datasets = load_dataset("json", data_files=data_files)
            train_ds = raw_datasets["train"].map(cls.format_instruction_prompt)
            eval_ds = raw_datasets["test"].map(cls.format_instruction_prompt)
            return train_ds, eval_ds

        # Synthetic fallback data generator for instruction tuning pipeline execution
        print("[!] Dataset files not found at specified paths. Using synthetic financial samples.")
        raw_data = [
            {
                "context": (
                    "Company X reported EBITDA margin expansion of 220bps to 18.5%, "
                    "but debt-to-equity ratio spiked to 4.2x following the acquisition."
                ),
                "ground_truth_answer": (
                    "Impact: NEGATIVE. While EBITDA margin expanded by 220bps to 18.5%, "
                    "the debt-to-equity ratio spike to 4.2x signals significant leverage risk."
                ),
            },
            {
                "context": (
                    "Company Y refinanced $500M senior unsecured notes at 4.5% interest, "
                    "extending debt maturity profile from 2026 to 2033."
                ),
                "ground_truth_answer": (
                    "Impact: POSITIVE. Refinancing $500M notes extends maturity horizon "
                    "by 7 years and mitigates near-term liquidity pressure."
                ),
            },
            {
                "context": (
                    "Company Z maintained net debt to EBITDA at 1.8x, with operating cash "
                    "flow flat year-over-year at $340M and capital expenditures decreasing 5%."
                ),
                "ground_truth_answer": (
                    "Impact: NEUTRAL. Stable leverage at 1.8x and steady operating cash flow "
                    "of $340M demonstrate consistent credit health with no immediate default risk."
                ),
            },
            {
                "context": (
                    "Company W suffered a rating downgrade from BBB- to BB+ (speculative grade) "
                    "due to persistent negative free cash flow of -$85M over three consecutive quarters."
                ),
                "ground_truth_answer": (
                    "Impact: NEGATIVE. Downgrade into speculative grade combined with -$85M "
                    "negative free cash flow significantly raises borrowing costs and refinancing risk."
                ),
            },
        ]
        df = pd.DataFrame(raw_data)
        dataset = Dataset.from_pandas(df)
        formatted_dataset = dataset.map(cls.format_instruction_prompt)

        # Split into train/validation
        split = formatted_dataset.train_test_split(test_size=0.5, seed=42)
        return split["train"], split["test"]
