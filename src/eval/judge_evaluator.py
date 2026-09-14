from typing import Any, Dict, List, Optional
from google import genai
from pydantic import BaseModel, Field
from src.config import settings


class JudgeScoreSchema(BaseModel):
    """Structured evaluation schema returned by the Gemini Judge."""

    financial_accuracy_score: float = Field(
        ...,
        description="Score between 0.0 and 1.0 assessing correctness of financial numbers and risk classification.",
    )
    hallucination_detected: bool = Field(
        ...,
        description="True if the candidate LLM invented ungrounded numbers, fabricated debt ratios, or false metrics.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed analytical breakdown explaining why the score was assigned.",
    )


class LLMJudgeEvaluator:
    """Uses Google Gemini (1.5 Pro) as an impartial LLM-as-a-Judge for financial analysis."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or settings.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please supply it in .env or pass it to LLMJudgeEvaluator."
            )
        self.client = genai.Client(api_key=key)

    def evaluate_model_output(
        self,
        context: str,
        model_output: str,
        ground_truth: str,
        model_name: str = "gemini-1.5-pro",
    ) -> JudgeScoreSchema:
        """
        Evaluates candidate model inference against ground truth and corporate disclosure context.
        """
        system_instruction = (
            "You are an impartial financial LLM evaluation benchmark. "
            "Compare the candidate model's answer against the context and ground truth. "
            "Assess whether all numerical values and risk classifications are exact and hallucination-free."
        )

        eval_prompt = f"""Context:
{context}

Ground Truth Answer:
{ground_truth}

Candidate Model Answer:
{model_output}

Evaluate accuracy, detect any hallucinations, and return structured evaluation details.
"""

        response = self.client.models.generate_content(
            model=model_name,
            contents=eval_prompt,
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": JudgeScoreSchema,
                "temperature": 0.0,
            },
        )

        return JudgeScoreSchema.model_validate_json(response.text)

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        model_name: str = "gemini-1.5-pro",
    ) -> List[Dict[str, Any]]:
        """
        Runs batch evaluation over a list of samples containing:
        - context
        - model_output
        - ground_truth
        """
        results = []
        for idx, sample in enumerate(samples):
            print(f"[+] Evaluating sample {idx + 1}/{len(samples)}...")
            score = self.evaluate_model_output(
                context=sample["context"],
                model_output=sample["model_output"],
                ground_truth=sample["ground_truth"],
                model_name=model_name,
            )
            results.append({
                "sample_index": idx,
                "context": sample["context"],
                "model_output": sample["model_output"],
                "ground_truth": sample["ground_truth"],
                "score": score.model_dump(),
            })
        return results
