"""Training loop for model #1 recursive masked reasoning diffuser."""

import copy
import torch
import torch.nn.functional as F

from config import tiny_config
from diffusion import cosine_mask_probability, make_mask, masked_diffusion_ce, remask_from_logits
from model import ReasoningDiffuser


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for avg, cur in zip(self.shadow.parameters(), model.parameters()):
            avg.mul_(self.decay).add_(cur, alpha=1.0 - self.decay)


def deep_supervision_step(model, batch, optimizer, ema=None):
    cfg = model.cfg
    x_ids, labels = batch['question'], batch['answer']
    x_mask = batch.get('question_mask')
    y, masked = make_mask(labels, 1.0, cfg.mask_id, cfg.pad_id)
    z = model.initial_state(x_ids.size(0), x_ids.device)
    total = torch.tensor(0.0, device=x_ids.device)

    optimizer.zero_grad(set_to_none=True)
    for step in range(cfg.supervision_steps):
        for _ in range(cfg.warmup_recursions):
            with torch.no_grad():
                logits, _, z, _ = model(x_ids, y, z, x_mask)
                y = remask_from_logits(logits, labels, masked, cosine_mask_probability(step, cfg.supervision_steps), cfg.mask_id, cfg.pad_id)
        logits, _, z, halt_logit = model(x_ids, y, z, x_mask)
        loss = masked_diffusion_ce(logits, labels, masked, cfg.pad_id, cfg.label_smoothing)
        correct = logits.argmax(-1).eq(labels).masked_fill(labels.eq(cfg.pad_id), True).all(dim=1).float()
        loss = loss + 0.05 * F.binary_cross_entropy_with_logits(halt_logit, correct)
        (loss / cfg.supervision_steps).backward()
        total = total + loss.detach()
        next_prob = cosine_mask_probability(step + 1, cfg.supervision_steps)
        y = remask_from_logits(logits.detach(), labels, masked, next_prob, cfg.mask_id, cfg.pad_id).detach()
        masked = y.eq(cfg.mask_id) & labels.ne(cfg.pad_id)
        z = z.detach()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if ema is not None:
        ema.update(model)
    return total / cfg.supervision_steps


def smoke_batch(cfg, batch_size=2):
    question = torch.randint(4, cfg.vocab_size, (batch_size, min(7, cfg.max_question_len)))
    answer = torch.randint(4, cfg.vocab_size, (batch_size, cfg.answer_len))
    answer[:, -1] = cfg.eos_id
    return {'question': question, 'question_mask': torch.ones_like(question, dtype=torch.bool), 'answer': answer}


if __name__ == '__main__':
    cfg = tiny_config(vocab_size=64, d_model=32, n_heads=4, d_ff=64, answer_len=6, recursion_depth=1, supervision_steps=3)
    model = ReasoningDiffuser(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    ema = EMA(model, cfg.ema_decay)
    loss = deep_supervision_step(model, smoke_batch(cfg), opt, ema)
    print(f'smoke loss: {loss.item():.4f}')
