import torch
from transformers import AutoModel, AutoTokenizer
from model import DSparkDraftModel
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Running smoke test on", device)

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-VL-450M", trust_remote_code=True)
base_model = AutoModel.from_pretrained("LiquidAI/LFM2.5-VL-450M", dtype=torch.bfloat16, trust_remote_code=True).to(device)
base_model.eval()

drafter = DSparkDraftModel().to(device)
optimizer = torch.optim.AdamW(drafter.parameters(), lr=1e-3)
lm_head_weight = base_model.language_model.embed_tokens.weight.float()

# Create mini dummy batch
input_ids = torch.randint(100, 5000, (2, 64), device=device)
with torch.no_grad():
    base_out = base_model(input_ids=input_ids, output_hidden_states=True)
    teacher_hidden = base_out.hidden_states[-1].float()

anchor_hidden = teacher_hidden[:, 10, :]
target_tokens = input_ids[:, 11:20]
prev_tokens = input_ids[:, 10:19]

draft_out = drafter(anchor_hidden, lm_head_weight, input_token_ids=prev_tokens)
base_logits = draft_out["base_logits"]
conf_logits = draft_out["confidence_logits"]
markov_logits = draft_out["markov_logits"]

loss_ce = F.cross_entropy(base_logits.view(-1, 65536), target_tokens.reshape(-1))
loss_markov = F.cross_entropy(markov_logits.view(-1, 65536), target_tokens.reshape(-1))
loss = loss_ce + 0.5 * loss_markov
loss.backward()
optimizer.step()

print("Loss computed successfully:", loss.item())
print("Smoke test passed successfully!")
