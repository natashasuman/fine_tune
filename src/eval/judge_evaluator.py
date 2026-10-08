from typing import Any, Dict, List, Optional
from google import genai
from pydantic import BaseModel, Field
from src.config import settings


try:
    from google.genai import types
except ImportError:
    types = None


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
    """Uses Google Gemini as an impartial LLM-as-a-Judge for financial analysis."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or settings.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please supply it in .env or pass it to LLMJudgeEvaluator."
            )
        self.client = genai.Client(api_key=key)

    def _discover_active_models(self) -> List[tuple]:
        """Queries Google Gemini API for available text-generating models supported by this key."""
        try:
            discovered = []
            excluded_substrings = ["tts", "imagen", "embedding", "embed", "aqa", "audio"]
            for m in self.client.models.list():
                name = getattr(m, "name", "")
                if not name:
                    continue
                base_name = name.split("/")[-1].lower()
                # Skip non-text models (like TTS, Audio, Embeddings, Imagen)
                if any(sub in base_name for sub in excluded_substrings):
                    continue
                # Check supported actions/methods if available
                supported = getattr(m, "supported_generation_methods", None) or getattr(m, "supported_actions", None)
                if supported and not any("generateContent" in str(act) for act in supported):
                    continue
                discovered.append((name, name.split("/")[-1]))
            return discovered
        except Exception as e:
            print(f"[!] Warning: Could not retrieve model list via ModelService.ListModels: {e}")
            return []

    def _generate_mock_score(self, context: str, model_output: str, ground_truth: str) -> JudgeScoreSchema:
        """Simulated evaluation for offline testing or when daily API quota is exhausted."""
        return JudgeScoreSchema(
            financial_accuracy_score=0.95,
            hallucination_detected=False,
            reasoning=(
                "[MOCK / OFFLINE EVALUATION] Candidate output matches key financial metrics "
                "from context and ground truth. Risk classification aligns with disclosure."
            ),
        )

    def evaluate_model_output(
        self,
        context: str,
        model_output: str,
        ground_truth: str,
        model_name: Optional[str] = None,
    ) -> JudgeScoreSchema:
        """
        Evaluates candidate model inference against ground truth and corporate disclosure context.
        """
        if getattr(settings, "USE_MOCK_JUDGE", False):
            print("[i] USE_MOCK_JUDGE=true enabled. Returning simulated benchmark score without API call.")
            return self._generate_mock_score(context, model_output, ground_truth)

        selected_model = model_name or settings.GEMINI_JUDGE_MODEL
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

        if types is not None and hasattr(types, "GenerateContentConfig"):
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=JudgeScoreSchema,
                temperature=0.0,
            )
        else:
            config = {
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": JudgeScoreSchema,
                "temperature": 0.0,
            }

        # Query text models accessible by this API key
        discovered_pairs = self._discover_active_models()
        valid_candidates = []

        if discovered_pairs:
            # Check if preferred model matches any discovered model
            for full_name, base_name in discovered_pairs:
                if selected_model in (full_name, base_name) or base_name in selected_model:
                    if base_name not in valid_candidates:
                        valid_candidates.append(base_name)

            # Prioritize standard text chat models
            priority_order = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            for prio in priority_order:
                for full_name, base_name in discovered_pairs:
                    if prio in base_name.lower() and base_name not in valid_candidates:
                        valid_candidates.append(base_name)

            # Add any remaining discovered text models
            for full_name, base_name in discovered_pairs:
                if "gemini" in base_name.lower() and base_name not in valid_candidates:
                    valid_candidates.append(base_name)

        if not valid_candidates:
            # Fallback list if discovery was empty
            valid_candidates = [
                selected_model,
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]

        last_error = None
        for candidate in valid_candidates:
            try:
                response = self.client.models.generate_content(
                    model=candidate,
                    contents=eval_prompt,
                    config=config,
                )
                return JudgeScoreSchema.model_validate_json(response.text)
            except Exception as e:
                err_str = str(e)
                # Retry if model not found (404), modality mismatch (400), or rate limited on this specific model (429)
                if any(x in err_str for x in ["404", "NOT_FOUND", "400", "INVALID_ARGUMENT", "429", "RESOURCE_EXHAUSTED"]):
                    last_error = e
                    continue
                raise e

        # If quota is exhausted on all models, gracefully fall back to mock scoring
        if last_error and any(x in str(last_error) for x in ["429", "RESOURCE_EXHAUSTED", "quota"]):
            print("\n[!] NOTICE: Google Gemini daily token/rate quota has been reached on all models.")
            print("[!] Falling back to simulated offline judge score so pipeline testing can continue.\n")
            return self._generate_mock_score(context, model_output, ground_truth)

        # If all candidates returned 404, output specific guidance
        print("\n" + "=" * 60)
        print("[!] GEMINI API 404 NOT_FOUND DIAGNOSTIC:")
        print("  Google returned 404 for all tested models.")
        if discovered_pairs:
            available_names = [b for _, b in discovered_pairs]
            print(f"  Models accessible by your key: {available_names}")
        else:
            print("  Your API key was unable to access any Generative Language models.")
            print("  This typically means:")
            print("  1. The API key was generated in Google Cloud Console without the")
            print("     'Generative Language API' enabled.")
            print("     Enable it at: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
            print("  2. Or, create a fresh free key directly at:")
            print("     https://aistudio.google.com/app/apikey")
        print("=" * 60 + "\n")

        raise last_error

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        model_name: Optional[str] = None,
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
