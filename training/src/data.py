"""
Dataset pipeline for DSpark drafter training.
Streams/batches datasets, tokenizes sequences, and feeds them into the training loop.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class DSparkTextDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.sequences = tokenized_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def prepare_dataset(tokenizer, max_length=128, total_samples=2400):
    print(f"Loading and assembling ~{total_samples} samples across domains...", flush=True)
    
    # 1. Code instruction
    try:
        code_data = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:600]")
        code_texts = [f"### Instruction:\n{item['query']}\n\n### Response:\n{item['answer']}" for item in code_data]
    except Exception as e:
        print("Falling back on python code instructions:", e, flush=True)
        code_data = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:600]")
        code_texts = [f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}" for item in code_data]

    # 2. Function calling / Tool use
    try:
        tool_data = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:600]")
        tool_texts = [item["chat"] for item in tool_data if "chat" in item]
    except Exception as e:
        print("Falling back on tool data:", e, flush=True)
        tool_texts = []

    # 3. Chat / SFT
    try:
        chat_data = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:600]")
        chat_texts = []
        for item in chat_data:
            if "messages" in item:
                dialogue = ""
                for msg in item["messages"]:
                    dialogue += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                chat_texts.append(dialogue)
    except Exception as e:
        print("Chat data fallback:", e, flush=True)
        chat_texts = []

    # 4. Math / CoT
    try:
        math_data = load_dataset("openai/gsm8k", "main", split="train[:600]")
        math_texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in math_data]
    except Exception as e:
        print("Math data fallback:", e, flush=True)
        math_texts = []

    all_texts = code_texts + tool_texts + chat_texts + math_texts
    print(f"Total raw text samples collected: {len(all_texts)}", flush=True)

    print("Tokenizing sequences...", flush=True)
    tokenized = []
    for text in all_texts:
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"][0]
        if len(tokens) >= 16:  # filter out trivially short sequences
            tokenized.append(tokens)

    print(f"Final valid tokenized sequences: {len(tokenized)}", flush=True)
    return DSparkTextDataset(tokenized)

def collate_fn(batch):
    # Dynamic padding to max sequence length in batch
    max_len = max(len(seq) for seq in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
        masks[i, :len(seq)] = True
    return {"input_ids": padded, "attention_mask": masks}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-VL-450M", trust_remote_code=True)
    dataset = prepare_dataset(tokenizer, max_length=256, total_samples=1000)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("Batch input_ids shape:", batch["input_ids"].shape)
    print("Batch mask shape:", batch["attention_mask"].shape)
