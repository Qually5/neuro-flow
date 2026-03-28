import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrapper_policy, size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict
import os
import math

# Configuration for distributed training
class TrainingConfig:
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "gpt2")
        self.dataset_path = os.getenv("DATASET_PATH", "./data/dummy_data.txt")
        self.epochs = int(os.getenv("EPOCHS", "3"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "2"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))
        self.warmup_steps = int(os.getenv("WARMUP_STEPS", "100"))
        self.gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
        self.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "512"))

# Dummy Dataset for demonstration
class DummyDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._load_and_process_data(data_path)

    def _load_and_process_data(self, data_path):
        # In a real scenario, load and tokenize actual text data
        print(f"[Rank {dist.get_rank()}] Loading and processing dummy data from {data_path}")
        dummy_texts = [
            "This is a sample sentence for fine-tuning a large language model.",
            "NeuroFlow provides efficient distributed training for LLMs.",
            "Scalable inference is crucial for real-world AI applications.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming various industries globally."
        ] * 20 # Make it longer for more lines
        
        tokenized_data = []
        for text in dummy_texts:
            encoded = tokenizer(text, max_length=self.max_seq_length, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_data.append({
                "input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze()
            })
        return tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DistributedTrainer:
    def __init__(self, config: TrainingConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Initialize distributed environment
        dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model and wrap with FSDP
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model = FSDP(model, auto_wrap_policy=transformer_auto_wrapper_policy) # Or use size_based_auto_wrap_policy
        
        self.dataset = DummyDataset(self.tokenizer, config.dataset_path, config.max_seq_length)
        self.dataloader = DataLoader(self.dataset, batch_size=config.batch_size, shuffle=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=len(self.dataloader) * config.epochs
        )

    def train(self):
        self.model.train()
        total_steps = len(self.dataloader) * self.config.epochs
        completed_steps = 0

        for epoch in range(self.config.epochs):
            self.dataloader.sampler.set_epoch(epoch) # For DistributedSampler
            print(f"[Rank {self.rank}] Epoch {epoch+1}/{self.config.epochs}")
            
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids = batch["input_ids"].to(self.rank)
                attention_mask = batch["attention_mask"].to(self.rank)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    completed_steps += 1
                    
                    if self.rank == 0 and completed_steps % 10 == 0:
                        print(f"[Rank {self.rank}] Step {completed_steps}/{total_steps}, Loss: {loss.item():.4f}")

        print(f"[Rank {self.rank}] Training complete.")
        dist.destroy_process_group()

if __name__ == "__main__":
    # Example usage: This script would typically be launched via torch.distributed.run
    # For local testing, you can simulate a single process
    # In a real distributed setup, rank and world_size would be provided by the launcher
    
    # Dummy setup for single GPU/process execution
    if os.environ.get("RANK") is None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    config = TrainingConfig()
    trainer = DistributedTrainer(config, rank, world_size)
    trainer.train()
