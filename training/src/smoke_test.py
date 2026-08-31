import torch
from transformers import AutoModel, AutoTokenizer
from model import DSparkMarkovModel
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Running smoke test on", device)

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-VL-450M", trust_remote_code=True)
base_model = AutoModel.from_pretrained("LiquidAI/LFM2.5-VL-450M", dtype=torch.bfloat16, trust_remote_code=True).to(device)
base_model.eval()

drafter = DSparkMarkovModel().to(device)
optimizer = torch.optim.AdamW(drafter.parameters(), lr=1e-3)
lm_head_weight = base_model.language_model.embed_tokens.weight.float()

# Create mini dummy batch
input_ids = torch.randint(100, 5000, (2, 64), device=device)
with torch.no_grad():
    base_out = base_model(input_ids=input_ids, output_hidden_states=True)
    teacher_hidden = base_out.hidden_states[-1].float()

target_tokens = input_ids[:, 11:20]
prev_tokens = input_ids[:, 10:19]
target_hiddens = [teacher_hidden] * len(drafter.target_layers)
draft_token_ids = torch.full_like(target_tokens, 64402)
draft_token_ids[:, 0] = input_ids[:, 10]

draft_out = drafter(
    target_layer_hiddens=target_hiddens,
    draft_token_ids=draft_token_ids,
    token_embd_weight=lm_head_weight,
    base_lm_head_weight=lm_head_weight,
    prev_tokens=prev_tokens,
)
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
