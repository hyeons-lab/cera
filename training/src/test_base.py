import torch
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer, AutoConfig

print("Testing base model config & tokenizer...")
model_id = "LiquidAI/LFM2.5-VL-450M"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print("Base config model_type:", getattr(config, 'model_type', 'unknown'))

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("Tokenizer vocab size:", len(tokenizer))

print("Loading base model with AutoModel...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
try:
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
except Exception as e:
    print("AutoModelForImageTextToText failed, trying AutoModel:", e)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

model.eval()
print("Base model loaded successfully on", device)
print("Model attributes:", [k for k in dir(model) if not k.startswith('_')][:15])
print("Language model type:", type(getattr(model, 'language_model', getattr(model, 'text_model', None))))
print("LM head weight shape:", model.get_output_embeddings().weight.shape if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None else "None")

# Test text-only forward pass
test_text = "The capital of France is Paris."
inputs = tokenizer(test_text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print("Outputs keys:", outputs.keys() if hasattr(outputs, 'keys') else dir(outputs))
if hasattr(outputs, 'logits'):
    print("Logits shape:", outputs.logits.shape)
if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
    print("Num hidden states:", len(outputs.hidden_states))
    print("Last hidden state shape:", outputs.hidden_states[-1].shape)

print("Test complete.")
