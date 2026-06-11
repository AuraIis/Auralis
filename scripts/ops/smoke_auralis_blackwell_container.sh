#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-auralis-blackwell}"
CHECKPOINT="${CHECKPOINT:-/workspace/v2data/checkpoints/runpod_import/pretrain_mix_v5_boosted_500m_a100_latest/best.pt}"
MODEL_CONFIG="${MODEL_CONFIG:-/workspace/v2data/configs/model/helix_v2_mid_500m_smart.yaml}"
TOKENIZER="${TOKENIZER:-/workspace/v2data/tokenizer/helix_v2_tokenizer.model}"

docker exec "$CONTAINER_NAME" bash -lc "
set -euo pipefail
cd /workspace/v2data
export AURALIS_USE_MAMBA_KERNEL=1
export AURALIS_USE_GLA_KERNEL=1
python - <<'PY'
from pathlib import Path
import torch
import sentencepiece as spm
from auralis.model import build_model
from auralis.model.backend_info import describe_model_backends

ckpt = Path('$CHECKPOINT')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), 'cap', torch.cuda.get_device_capability(0))

model = build_model('$MODEL_CONFIG').to('cuda')
print('backends', describe_model_backends(model)['summary'])

payload = torch.load(ckpt, map_location='cuda', weights_only=False)
state = {k.replace('_orig_mod.', ''): v for k, v in payload['model'].items()}
missing, extra = model.load_state_dict(state, strict=False)
print('state_dict', 'missing', len(missing), 'extra', len(extra))

sp = spm.SentencePieceProcessor(model_file='$TOKENIZER')
ids = sp.EncodeAsIds('Berlin ist die Hauptstadt von')
x = torch.tensor([ids], dtype=torch.long, device='cuda')
model.eval()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(input_ids=x)
next_id = int(out['logits'][0, -1].argmax().item())
print('forward', tuple(out['logits'].shape), 'finite', torch.isfinite(out['logits']).all().item(), 'next', sp.DecodeIds([next_id]))
PY
"
