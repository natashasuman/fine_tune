import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from src.config import settings
from src.finetune.dataset_loader import FinancialDatasetPreparer


class QLoRAFinancialTrainer:
    """Trainer engine that configures 4-bit NF4 Quantization and trains LoRA adapters."""

    def __init__(self, custom_settings=None):
        self.settings = custom_settings or settings
        self.cuda_available = torch.cuda.is_available()

        # 4-bit Quantization Config (NF4 with double quantization)
        if self.cuda_available:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            self.bnb_config = None

    def train(self):
        print(f"[+] Loading base model: {self.settings.BASE_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.BASE_MODEL_ID,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
        }

        if self.cuda_available and self.bnb_config is not None:
            print("[+] Applying 4-bit NF4 bitsandbytes quantization...")
            model_kwargs["quantization_config"] = self.bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            print("[!] CUDA not detected or bitsandbytes 4-bit disabled. Loading in default precision.")
            model_kwargs["device_map"] = "auto" if self.cuda_available else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self.settings.BASE_MODEL_ID,
            **model_kwargs,
        )

        if self.cuda_available:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA Configuration
        peft_config = LoraConfig(
            r=self.settings.LORA_R,
            lora_alpha=self.settings.LORA_ALPHA,
            target_modules=self.settings.TARGET_MODULES,
            lora_dropout=self.settings.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        print("[+] Trainable parameters status:")
        model.print_trainable_parameters()

        # Load datasets
        train_ds, eval_ds = FinancialDatasetPreparer.load_and_preprocess(
            train_path=self.settings.TRAIN_DATA_PATH,
            eval_path=self.settings.EVAL_DATA_PATH,
        )

        os.makedirs(self.settings.OUTPUT_DIR, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.settings.OUTPUT_DIR,
            per_device_train_batch_size=self.settings.BATCH_SIZE,
            gradient_accumulation_steps=self.settings.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.settings.LEARNING_RATE,
            num_train_epochs=self.settings.NUM_EPOCHS,
            logging_steps=10,
            fp16=self.cuda_available,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_ds is not None else "no",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=self.settings.MAX_SEQ_LENGTH,
            tokenizer=tokenizer,
            args=training_args,
        )

        print("[+] Starting QLoRA Fine-Tuning execution...")
        trainer.train()

        print(f"[+] Saving LoRA adapter weights and tokenizer to {self.settings.OUTPUT_DIR}")
        trainer.model.save_pretrained(self.settings.OUTPUT_DIR)
        tokenizer.save_pretrained(self.settings.OUTPUT_DIR)
        print("[+] Fine-tuning completed successfully.")
