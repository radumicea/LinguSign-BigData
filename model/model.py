import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers.models.t5 import T5ForConditionalGeneration, T5Tokenizer

with open('config.json') as f:
    config = json.load(f)


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device)[None, :]
    return idx < lengths[:, None]


class TemporalConvStack(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x_ = x.transpose(1, 2)
        x_ = F.relu(self.bn1(self.conv1(x_)))
        x_ = self.pool(x_)
        x_ = F.relu(self.bn2(self.conv2(x_)))
        x_ = self.pool(x_)
        y = x_.transpose(1, 2)
        y = self.ln(y)

        new_lengths = torch.div(lengths, 4, rounding_mode='floor').clamp(min=1)
        return y, new_lengths


class CrossModalMLP(nn.Module):
    def __init__(self, dim: int, dropout: float, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class SignAdapterCore(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            CrossModalMLP(out_dim, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttentionPool(nn.Module):
    """
    Content-aware pooling over HxW to a single 1024-d vector per window.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        # 1x1 conv to produce an attention score per spatial location
        self.attn_conv = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, 1, H, W] from I3D mixed_5c
        returns: [B, T, C]
        """
        B, T, C, _, H, W = x.shape
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]

        scores = self.attn_conv(x)  # [B*T, 1, H, W]
        scores = scores.view(B * T, -1)  # [B*T, H*W]
        attn = torch.softmax(scores, dim=-1).view(B * T, 1, H * W)  # [B*T,1,HW]

        feats = x.view(B * T, C, H * W)  # [B*T, C, HW]
        pooled = torch.bmm(attn, feats.transpose(1, 2))  # [B*T,1,C]
        pooled = pooled.squeeze(1)  # [B*T, C]

        return pooled.view(B, T, C)  # [B, T, C]


class SignAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, max_vis_tokens, dropout=0.1):
        super().__init__()
        self.spatial_pool = SpatialAttentionPool(in_dim)
        self.core = SignAdapterCore(in_dim, out_dim, dropout)
        self.temporal = TemporalConvStack(out_dim)
        self.pos_emb = nn.Embedding(max_vis_tokens, out_dim)

    def forward(
        self, vis: torch.Tensor, vis_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.spatial_pool(vis)
        x = self.core(x)

        x, new_lengths = self.temporal(x, vis_lengths)

        semantic = x

        B, T_prime, _ = x.shape
        pos_ids = torch.arange(T_prime, device=x.device).unsqueeze(0).expand(B, T_prime)
        x = x + self.pos_emb(pos_ids)

        return x, semantic, new_lengths


class LinguSign(nn.Module):
    def __init__(
        self,
        max_vis_tokens: int = config['max_vis_tokens'],
        max_text_tokens: int = config['max_text_tokens'],
        mt5_model_name: str = config['mt5_model_name'],
        lora_r: int = config['lora_r'],
        lora_alpha: int = config['lora_alpha'],
        lora_dropout: float = config['lora_dropout'],
        prefix_text: str = config['prefix_text'],
        label_smoothing: float = config['label_smoothing'],
        alpha_vt: float = config['alpha_vt'],
        combined_loss: bool = config['combined_loss'],
    ):
        super().__init__()
        self.max_vis_tokens = max_vis_tokens
        self.max_text_tokens = max_text_tokens
        self.prefix_text = prefix_text
        self.alpha_vt = float(alpha_vt)
        self.combined_loss = combined_loss

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        self.tokenizer = T5Tokenizer.from_pretrained(mt5_model_name, legacy=False)

        lcfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type=TaskType.SEQ_2_SEQ_LM,
            bias='none',
            target_modules=['q', 'v'],
        )
        base = T5ForConditionalGeneration.from_pretrained(
            mt5_model_name,
            torch_dtype=self.dtype,
        )
        self.mt5 = get_peft_model(base, lcfg).to(self.device)

        hidden = self.mt5.config.d_model  # type: ignore
        self.sign_adapter = SignAdapter(
            in_dim=1024, out_dim=hidden, max_vis_tokens=self.max_vis_tokens
        ).to(self.device, self.dtype)  # type: ignore

        self.ignore_index = -100
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            label_smoothing=label_smoothing,
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(2.6592, device=self.device, dtype=self.dtype)
        )

        pref = self.tokenizer(
            self.prefix_text,
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_tokens,
        )
        self.register_buffer('prefix_input_ids', pref['input_ids'].to(self.device))
        self.register_buffer('prefix_attn_mask', pref['attention_mask'].to(self.device))

    def _get_prefix_embeds(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pref_ids = self.prefix_input_ids.expand(bsz, -1)  # type: ignore
        pref_mask = self.prefix_attn_mask.expand(bsz, -1)  # type: ignore
        with torch.no_grad():
            pref_emb = self.mt5.get_encoder().embed_tokens(pref_ids)  # type: ignore
        pref_emb = pref_emb.to(self.dtype)
        return pref_emb, pref_mask

    def _enter_warmup(self):
        for p in self.mt5.parameters():
            p.requires_grad = False
        for p in self.sign_adapter.parameters():
            p.requires_grad = True
        self.logit_scale.requires_grad = True

    def _enter_joint(self):
        for n, p in self.mt5.named_parameters():
            if ('lora_A' in n) or ('lora_B' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        for p in self.sign_adapter.parameters():
            p.requires_grad = True
        self.logit_scale.requires_grad = True

    def vt_align_loss(
        self,
        vis_seq: torch.Tensor,
        vis_mask: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = vis_mask.unsqueeze(-1).to(vis_seq.dtype)
        v = (vis_seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        v = F.normalize(v, dim=-1, eps=1e-6)

        with torch.no_grad():
            enc_out = self.mt5.encoder(
                input_ids=text_ids, attention_mask=text_mask
            ).last_hidden_state  # type: ignore
        t_m = text_mask.unsqueeze(-1).to(enc_out.dtype)
        t = (enc_out * t_m).sum(dim=1) / t_m.sum(dim=1).clamp(min=1.0)
        t = F.normalize(t, dim=-1, eps=1e-6)

        clamped_logit_scale = self.logit_scale.clamp(0, 5)

        sims = clamped_logit_scale.exp() * (t @ v.t())
        labels = torch.arange(sims.size(0), device=sims.device)
        loss1 = F.cross_entropy(sims, labels)
        loss2 = F.cross_entropy(sims.t(), labels)
        return 0.5 * (loss1 + loss2)

    def forward(
        self,
        vis_tokens: torch.Tensor,
        vis_lengths: torch.Tensor,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        in_warmup: bool,
    ) -> Dict[str, torch.Tensor]:
        B, T_raw, C, _, _, _ = vis_tokens.shape
        assert C == 1024

        vis_tokens = vis_tokens.to(self.device, self.dtype)
        vis_lengths = vis_lengths.to(self.device)
        text_ids = text_ids.to(self.device)
        text_lengths = text_lengths.to(self.device)

        T = min(T_raw, self.max_vis_tokens)

        vis_lengths_clamped = vis_lengths.clamp(max=T)

        pref_emb, pref_mask = self._get_prefix_embeds(B)
        x_sa, x_semantic, vis_lengths_new = self.sign_adapter(
            vis_tokens[:, :T], vis_lengths_clamped
        )
        T_prime = x_sa.size(1)
        vis_mask = lengths_to_mask(vis_lengths_new.clamp(max=T_prime), T_prime)

        inputs_embeds = torch.cat([pref_emb, x_sa], dim=1)
        attn_mask = torch.cat([pref_mask, vis_mask], dim=1).long()

        L = text_ids.size(1)
        text_lengths = text_lengths.clamp(max=L)
        text_mask = lengths_to_mask(text_lengths, L).long()

        labels = text_ids.clone()
        labels[text_mask == 0] = self.ignore_index

        if in_warmup:
            self._enter_warmup()
            vt = self.vt_align_loss(x_semantic, vis_mask, text_ids, text_mask)
            return {'loss': vt, 'vt_loss': vt}

        self._enter_joint()

        out = self.mt5(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True,
        )

        ce = self.criterion(
            out.logits.view(-1, out.logits.size(-1)),
            labels.view(-1),
        )

        if self.combined_loss:
            vt = self.vt_align_loss(x_semantic, vis_mask, text_ids, text_mask)
            loss = ce + self.alpha_vt * vt
            return {'loss': loss, 'ce_loss': ce, 'vt_loss': vt}
        else:
            return {'loss': ce, 'ce_loss': ce}

    @torch.no_grad()
    def generate(
        self,
        vis_tokens: torch.Tensor,
        vis_lengths: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.5,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 4,
    ) -> List[str]:
        vis_tokens = vis_tokens.to(self.device, self.dtype)
        vis_lengths = vis_lengths.to(self.device)

        B, T_raw, _, _, _, _ = vis_tokens.shape
        T = min(T_raw, self.max_vis_tokens)
        vis_lengths_clamped = vis_lengths.clamp(max=T)

        pref_emb, pref_mask = self._get_prefix_embeds(B)
        x_sa, _, vis_lengths_new = self.sign_adapter(
            vis_tokens[:, :T], vis_lengths_clamped
        )
        T_prime = x_sa.size(1)
        vis_mask = lengths_to_mask(vis_lengths_new.clamp(max=T_prime), T_prime)

        inputs_embeds = torch.cat([pref_emb, x_sa], dim=1)
        attn_mask = torch.cat([pref_mask, vis_mask], dim=1).long()

        ids = self.mt5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
        )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)
