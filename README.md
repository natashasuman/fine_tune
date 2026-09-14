# Financial Credit-Risk QLoRA Fine-Tuning & Evaluation Pipeline

A production-grade PyTorch pipeline that fine-tunes an open-weight base model (`Qwen/Qwen2.5-7B-Instruct`) on financial credit-risk disclosures using 4-bit NF4 Quantized Low-Rank Adaptation (QLoRA via PEFT & bitsandbytes), with automated LLM-as-a-Judge evaluation powered by Google Gemini 1.5 Pro.

---

## 📁 Repository Structure

```text
fine_tune/
├── data/
│   ├── train_dataset.jsonl        # Training financial disclosure records
│   └── eval_dataset.jsonl         # Evaluation benchmark records
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic Settings & Hyperparameters
│   ├── finetune/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py      # ChatML dataset formatting & preprocessing
│   │   └── trainer.py             # 4-bit NF4 QLoRA SFTTrainer engine
│   ├── eval/
│   │   ├── __init__.py
│   │   └── judge_evaluator.py     # Gemini 1.5 Pro LLM-as-a-Judge benchmark
│   └── serve/
│       ├── __init__.py
│       └── predictor.py           # FastAPI serving and PEFT model inference
├── .env.example                   # Environment variable template
├── main.py                        # Pipeline entrypoint CLI (train, eval, serve, demo)
├── prd.md                         # Product Requirements Document
├── requirements.txt               # Dependency specification
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository and navigate to the folder:**
   ```bash
   cd C:\BetaProj\Project\fine_tune
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and configure your keys:
   ```bash
   cp .env.example .env
   ```
   Set your Google Gemini API key:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 🚀 Usage

### 1. Run Default Demo / Judge Benchmark
Runs the LLM-as-a-Judge evaluation sample using Gemini 1.5 Pro:
```bash
python main.py
# or
python main.py --mode demo
```

### 2. Fine-Tune with QLoRA
Executes 4-bit NF4 quantized LoRA fine-tuning using `trl.SFTTrainer`:
```bash
python main.py --mode train
```
Adapter weights and tokenizer are saved into `./lora_financial_credit_weights`.

### 3. Serve via FastAPI
Launch the REST API for model inference:
```bash
python main.py --mode serve
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

#### Sample Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "context": "Company X reported EBITDA margin expansion of 220bps to 18.5%, but debt-to-equity ratio spiked to 4.2x following the acquisition.",
       "max_new_tokens": 256,
       "temperature": 0.1
     }'
```

---

## ⚙️ Configuration & Hyperparameters

Key settings in [src/config.py](file:///C:/BetaProj/Project/fine_tune/src/config.py) (override via `.env`):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `BASE_MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | Base model Hugging Face repository ID |
| `OUTPUT_DIR` | `./lora_financial_credit_weights` | Path to save trained LoRA adapters |
| `LORA_R` | `16` | Rank parameter for LoRA adapters |
| `LORA_ALPHA` | `32` | Scaling parameter for LoRA updates |
| `LORA_DROPOUT` | `0.05` | Dropout probability for LoRA layers |
| `BATCH_SIZE` | `2` | Training batch size per device |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Effective batch size multiplier |
| `LEARNING_RATE` | `2e-4` | Peak learning rate for AdamW |
| `NUM_EPOCHS` | `3` | Number of training epochs |
| `MAX_SEQ_LENGTH` | `1024` | Maximum sequence length for SFT |
