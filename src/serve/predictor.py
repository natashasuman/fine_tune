import os
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import settings

app = FastAPI(
    title="Financial Credit-Risk QLoRA Inference API",
    description="Production API serving fine-tuned credit-risk and financial liquidity analysis models.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    context: str = Field(
        ...,
        description="Corporate disclosure or earnings text for credit risk analysis.",
        example="Company X reported EBITDA margin expansion of 220bps to 18.5%, but debt-to-equity ratio spiked to 4.2x following the acquisition.",
    )
    max_new_tokens: Optional[int] = Field(256, ge=16, le=1024)
    temperature: Optional[float] = Field(0.1, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    context: str
    assessment: str
    model_id: str
    adapter_applied: bool


class FinancialPredictor:
    """Inference engine for fine-tuned LoRA models."""

    def __init__(
        self,
        base_model_id: str = settings.BASE_MODEL_ID,
        adapter_path: Optional[str] = settings.OUTPUT_DIR,
    ):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_available else "cpu"

        print(f"[+] Initializing FinancialPredictor with base: {base_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.cuda_available else torch.float32,
            "device_map": "auto" if self.cuda_available else None,
        }
        self.model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

        # Attach LoRA adapter if present
        self.adapter_applied = False
        if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")) or \
           (adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))):
            print(f"[+] Loading PEFT adapter from {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.adapter_applied = True
        else:
            print("[!] No trained LoRA adapter found in output directory; using base model directly.")

        self.model.eval()

    def generate_assessment(
        self,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        system_prompt = (
            "You are a Senior Financial Risk Analyst. Analyze corporate disclosures, "
            "determine credit risk impact (POSITIVE, NEGATIVE, NEUTRAL), and extract financial metrics."
        )
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nCorporate Disclosure:\n{context}\n\n"
            f"Task: Assess liquidity risk and state impact.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.cuda_available:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated assistant tokens
        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        return response_text


# Global predictor instance (lazy initialized)
_predictor: Optional[FinancialPredictor] = None


def get_predictor() -> FinancialPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FinancialPredictor()
    return _predictor


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "financial-qlora-inference",
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predictor = get_predictor()
        assessment = predictor.generate_assessment(
            context=request.context,
            max_new_tokens=request.max_new_tokens or 256,
            temperature=request.temperature or 0.1,
        )
        return PredictionResponse(
            context=request.context,
            assessment=assessment,
            model_id=predictor.base_model_id,
            adapter_applied=predictor.adapter_applied,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
