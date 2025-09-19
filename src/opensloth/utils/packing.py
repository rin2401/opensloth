# utils/packing.py
from typing import List, Dict, Any, Optional, Tuple
import torch
from typing import Any, Dict, List, Union, Optional
import torch

def _to_list_of_tensors(x) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [row.clone() for row in x]
    return [torch.as_tensor(item).clone() for item in x]

def _trim_by_mask(ids: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None or mask.dtype == torch.bool:
        L = int(mask.sum().item()) if mask is not None else ids.size(0)
    else:  # int mask
        L = int(mask.sum().item())
    return ids[:L]

def _chunk(ids: torch.Tensor, lbls: torch.Tensor, cap: int):
    out = []
    s = 0
    while s < ids.size(0):
        e = min(s + cap, ids.size(0))
        out.append((ids[s:e], lbls[s:e]))
        s = e
    return out

def _best_fit_decreasing(lengths: List[int], cap: int) -> List[List[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    bins, space = [], []
    for i in order:
        l = lengths[i]
        best, best_left = -1, cap + 1
        for b, left in enumerate(space):
            if l <= left and (left - l) < best_left:
                best, best_left = b, left - l
        if best == -1:
            bins.append([i]); space.append(cap - l)
        else:
            bins[best].append(i); space[best] -= l
    return bins
def _write_segment_block(blk_row: torch.Tensor, start: int, length: int, mode: str = "tri"):
    """
    Write a segment attention pattern into blk_row[start:start+length, start:start+length].

    mode:
      - "tri": lower-triangular (causal LM; recommended)
      - "diag": identity (only the main diagonal is True)
      - "dense": full ones (legacy behavior)
    """
    sl = slice(start, start + length)
    if mode == "tri":
        pat = torch.tril(torch.ones((length, length), dtype=torch.bool, device=blk_row.device))
    elif mode == "diag":
        pat = torch.eye(length, dtype=torch.bool, device=blk_row.device)
    elif mode == "dense":
        pat = torch.ones((length, length), dtype=torch.bool, device=blk_row.device)
    else:
        raise ValueError(f"Unknown mode={mode!r}; use 'tri' | 'diag' | 'dense'")
    blk_row[sl, sl] = pat
    
def pack(
    batch: Dict[str, Any],
    max_seql: int = 64,
    pad_token_id: Optional[int] = None,
    pad_label_id: int = -100,
    device: Optional[torch.device] = None,
    return_token_mask: bool = True,
    attn_mode='tri'
) -> Dict[str, torch.Tensor]:
    """
    Packs (B,L) into P rows via BFD with capacity max_seql, then trims width to:
        L_eff = min(max_seql, max(packed_lengths))
    Returns block-diagonal attention (P, L_eff, L_eff) and per-row token mask (P, L_eff).
    Required keys in `batch`: input_ids, attention_mask, labels.
    """
    ids_list    = _to_list_of_tensors(batch["input_ids"])
    mask_list   = _to_list_of_tensors(batch.get("attention_mask")) if "attention_mask" in batch else [None]*len(ids_list)
    labels_list = _to_list_of_tensors(batch["labels"])
    B = len(ids_list)

    if pad_token_id is None:
        # try to infer from padded rows if any; else default 0
        pad_token_id = 0
        for ids, m in zip(ids_list, mask_list):
            if m is not None and not bool(m.all()):
                pad_idx = int(m.sum().item())
                pad_token_id = int(ids[pad_idx].item())
                break

    # trim by mask & chunk longer-than-capacity
    seq_ids, seq_lbls = [], []
    for ids, m, lbl in zip(ids_list, mask_list, labels_list):
        ids_t = _trim_by_mask(ids, m) if m is not None else ids
        lbl_t = _trim_by_mask(lbl, m) if m is not None else lbl
        if ids_t.size(0) > max_seql:
            seq = _chunk(ids_t, lbl_t, max_seql)
            for a,b in seq: seq_ids.append(a); seq_lbls.append(b)
        else:
            seq_ids.append(ids_t); seq_lbls.append(lbl_t)

    lengths = [int(t.size(0)) for t in seq_ids]
    if not lengths:
        raise ValueError("No sequences to pack.")

    bins = _best_fit_decreasing(lengths, max_seql)

    row_ids, row_lbls, row_pos, row_blk, row_used = [], [], [], [], []
    for bin_idxs in bins:
        segs = [(seq_ids[i], seq_lbls[i]) for i in bin_idxs]
        seg_lens = [s[0].size(0) for s in segs]
        used = sum(seg_lens)

        ids_row  = torch.full((max_seql,), pad_token_id, dtype=torch.long)
        lbl_row  = torch.full((max_seql,), pad_label_id, dtype=torch.long)
        pos_row  = torch.zeros((max_seql,), dtype=torch.long)
        blk_row  = torch.zeros((max_seql, max_seql), dtype=torch.bool)

        cur = 0
        for (ids_seg, lbl_seg), L in zip(segs, seg_lens):
            end = cur + L
            ids_row[cur:end] = ids_seg
            lbl_row[cur:end] = lbl_seg
            pos_row[cur:end] = torch.arange(L, dtype=torch.long)
            _write_segment_block(blk_row, cur, L, mode=attn_mode)
            cur = end

        row_ids.append(ids_row)
        row_lbls.append(lbl_row)
        row_pos.append(pos_row)
        row_blk.append(blk_row)
        row_used.append(used)

    # stack at capacity
    input_ids      = torch.stack(row_ids, 0)       # (P, max_seql)
    labels         = torch.stack(row_lbls, 0)      # (P, max_seql)
    position_ids   = torch.stack(row_pos, 0)       # (P, max_seql)
    attention_mask = torch.stack(row_blk, 0)       # (P, max_seql, max_seql)
    used_lengths   = torch.tensor(row_used, dtype=torch.long)

    # ---- TRIM WIDTH to L_eff = min(max_seql, max(used_lengths)) ----
    L_eff = int(min(max_seql, int(used_lengths.max().item())))
    if L_eff < max_seql:
        input_ids      = input_ids[:, :L_eff]
        labels         = labels[:, :L_eff]
        position_ids   = position_ids[:, :L_eff]
        attention_mask = attention_mask[:, :L_eff, :L_eff]

    if device is not None:
        input_ids      = input_ids.to(device)
        labels         = labels.to(device)
        position_ids   = position_ids.to(device)
        attention_mask = attention_mask.to(device)
        used_lengths   = used_lengths.to(device)

    out = {
        "input_ids": input_ids,                   # (P, L_eff)
        "labels": labels,                         # (P, L_eff)
        "position_ids": position_ids,             # (P, L_eff)
        "attention_mask": attention_mask,         # (P, L_eff, L_eff) block-diagonal
        "packed_lengths": used_lengths,           # (P,)
    }
    if return_token_mask:
        # 1D per-row token mask (P, L_eff) for convenience
        P, L_eff = input_ids.shape
        token_mask = torch.zeros((P, L_eff), dtype=torch.long, device=input_ids.device)
        for i in range(P):
            token_mask[i, :int(used_lengths[i].item())] = 1
        out["token_mask"] = token_mask
    return out



def packed_collator(features, tokenizer):
    batch = {k: [f[k] for f in features] for k in features[0]}
    return pack(
        batch,
        max_seql=128,
        pad_token_id=tokenizer.pad_token_id,
        pad_label_id=-100,
        causal=True,
    )