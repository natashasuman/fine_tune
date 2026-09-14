import argparse
import sys
from src.config import settings
from src.eval.judge_evaluator import LLMJudgeEvaluator
from src.finetune.trainer import QLoRAFinancialTrainer


def run_pipeline():
    print("=== FINANCIAL QLORA FINE-TUNING & EVALUATION ENGINE ===")

    # 1. Simulate Judge Evaluation run on model output
    context_sample = (
        "Company Z revenue fell 12% YoY, but net debt was reduced by $120M using free cash flow."
    )
    ground_truth = (
        "Impact: NEUTRAL/STABLE. Top-line revenue declined by 12%, but balance sheet "
        "leverage improved via $120M debt reduction."
    )
    candidate_output = (
        "Impact: NEUTRAL. Revenue dropped 12% YoY, but debt decreased by $120M from cash flow."
    )

    print("\n[+] Running Gemini 1.5 Pro LLM-as-a-Judge Evaluation Benchmark...")
    try:
        evaluator = LLMJudgeEvaluator()
        eval_result = evaluator.evaluate_model_output(
            context=context_sample,
            model_output=candidate_output,
            ground_truth=ground_truth,
        )

        print("\n--- EVALUATION BENCHMARK METRICS ---")
        print(f"Financial Accuracy Score: {eval_result.financial_accuracy_score * 100:.1f}%")
        print(f"Hallucination Detected:   {eval_result.hallucination_detected}")
        print(f"Judge Reasoning:          {eval_result.reasoning}")
        print("====================================================")
    except Exception as e:
        print(f"[!] Evaluation could not complete: {e}")
        print("[!] Verify GEMINI_API_KEY is configured in your .env file.")


def run_training():
    print("=== STARTING QLORA FINE-TUNING PIPELINE ===")
    trainer = QLoRAFinancialTrainer()
    trainer.train()


def run_server():
    import uvicorn
    print(f"=== STARTING INFERENCE SERVER ON {settings.HOST}:{settings.PORT} ===")
    uvicorn.run("src.serve.predictor:app", host=settings.HOST, port=settings.PORT, reload=False)


def main():
    parser = argparse.ArgumentParser(
        description="Financial Credit-Risk QLoRA Pipeline & Evaluation Suite"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "eval", "serve"],
        default="demo",
        help="Pipeline execution mode: 'demo' (evaluation sample), 'train' (QLoRA SFT), 'eval', or 'serve' (FastAPI).",
    )
    args = parser.parse_args()

    if args.mode in ("demo", "eval"):
        run_pipeline()
    elif args.mode == "train":
        run_training()
    elif args.mode == "serve":
        run_server()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_pipeline()
