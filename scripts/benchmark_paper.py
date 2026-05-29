"""
Benchmark script for CausalWanSelfAttention with controllable token reuse ratio.

This script implements the ultimate 'MotionCache' inspired architecture:
1. All Q/K/V are mapped to 1D Hilbert space upfront.
2. Lightweight L1 Frame-Difference + Alpha Soft-mapping replaces heavy Neighbor Search.
3. Token Accumulators drive the computation mask.
4. In-place Token Copy replaces expensive Motion Compensation Gather.
5. Restore to 2D Raster happens only at the very end of the layer.
"""

import argparse
import math
import time
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fallback attention
# ---------------------------------------------------------------------------
try:
    from wan.modules.attention import attention
except Exception:
    def attention(q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


def get_hilbert_mapping_indices(H: int, W: int, device='cpu'):
    n = 1
    while n < max(H, W):
        n *= 2
    num_tokens = H * W
    hilbert_to_raster = torch.zeros(num_tokens, dtype=torch.long, device=device)
    raster_to_hilbert = torch.zeros(num_tokens, dtype=torch.long, device=device)

    def _hilbert_d2xy(n, d):
        t = d; x = y = 0; s = 1
        while s < n:
            rx = 1 & (t // 2); ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1: x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x, y = x + s * rx, y + s * ry
            t //= 4; s *= 2
        return x, y

    valid_idx = 0
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if x < W and y < H:
            raster_idx = y * W + x
            hilbert_to_raster[valid_idx] = raster_idx
            raster_to_hilbert[raster_idx] = valid_idx
            valid_idx += 1
    return hilbert_to_raster, raster_to_hilbert


# ---------------------------------------------------------------------------
# Asynchronous Fine-grained Timer
# ---------------------------------------------------------------------------
class CUDATimer:
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
        self._events = []
        self._cpu_times = []

    def start(self, name: str):
        if torch.cuda.is_available():
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            self._events.append({"name": name, "start": evt, "end": None})
        else:
            self._cpu_times.append({"name": name, "start": time.perf_counter(), "end": None})

    def end(self):
        if torch.cuda.is_available():
            evt = torch.cuda.Event(enable_timing=True)
            evt.record()
            for e in reversed(self._events):
                if e["end"] is None:
                    e["end"] = evt
                    break
        else:
            now = time.perf_counter()
            for e in reversed(self._cpu_times):
                if e["end"] is None:
                    e["end"] = now
                    break

    def flush(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            run_sums = {}
            for e in self._events:
                if e["end"] is not None:
                    ms = e["start"].elapsed_time(e["end"])
                    run_sums[e["name"]] = run_sums.get(e["name"], 0.0) + ms
            for k, v in run_sums.items():
                self.times.setdefault(k, []).append(v)
            self._events.clear()
        else:
            run_sums = {}
            for e in self._cpu_times:
                if e["end"] is not None:
                    ms = (e["end"] - e["start"]) * 1000.0
                    run_sums[e["name"]] = run_sums.get(e["name"], 0.0) + ms
            for k, v in run_sums.items():
                self.times.setdefault(k, []).append(v)
            self._cpu_times.clear()

    def reset(self):
        self.times.clear()
        self._events.clear()
        self._cpu_times.clear()


# ---------------------------------------------------------------------------
# Benchmark Attention Module
# ---------------------------------------------------------------------------
class BenchmarkCausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)

        h_to_r, r_to_h = get_hilbert_mapping_indices(30, 52)
        self.register_buffer("hilbert_to_raster", h_to_r, persistent=False)
        self.register_buffer("raster_to_hilbert", r_to_h, persistent=False)

    def forward(
        self, x, seq_lens, grid_sizes, freqs, kv_cache, current_start=0,
        cache_start=None, injected_reuse_mask: Optional[torch.Tensor] = None,
        timer: Optional[CUDATimer] = None
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None: cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q_raster, k_raster, v_raster = qkv_fn(x)

        # ---------------------------------------------------------------
        # Rope & Global Hilbert Mapping
        # ---------------------------------------------------------------
        if timer: timer.start("rope_and_hilbert")
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        f = q_raster.shape[1] // frame_seqlen
        current_start_frame = current_start // frame_seqlen
        
        roped_query_raster = causal_rope_apply(q_raster, grid_sizes, freqs, start_frame=current_start_frame).type_as(v_raster)
        roped_key_raster = causal_rope_apply(k_raster, grid_sizes, freqs, start_frame=current_start_frame).type_as(v_raster)

        # 将所有的特征强行拉入 1D Hilbert 空间！从这里开始，告别 2D！
        # 使用 hilbert_to_raster 从原图中提取像素排成一维
        q = q_raster.view(b, f, frame_seqlen, n, d)[:, :, self.hilbert_to_raster, :, :].view(b, s, n, d)
        roped_query = roped_query_raster.view(b, f, frame_seqlen, n, d)[:, :, self.hilbert_to_raster, :, :].view(b, s, n, d)
        roped_key = roped_key_raster.view(b, f, frame_seqlen, n, d)[:, :, self.hilbert_to_raster, :, :].view(b, s, n, d)
        v = v_raster.view(b, f, frame_seqlen, n, d)[:, :, self.hilbert_to_raster, :, :].view(b, s, n, d)
        if timer: timer.end()

        # ---------------------------------------------------------------
        # KV Cache Update (Always stores in Hilbert order now!)
        # ---------------------------------------------------------------
        if timer: timer.start("kv_cache_update")
        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]

        local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index -= num_evicted_tokens

        local_start_index = local_end_index - num_new_tokens
        kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        kv_cache["v"][:, local_start_index:local_end_index] = v
        k_cache_sliced = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
        v_cache_sliced = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
        if timer: timer.end()

        # ---------------------------------------------------------------
        # MotionCache - Lightweight Decision Logic
        # ---------------------------------------------------------------
        C = n * d
        N = frame_seqlen
        mask_1d = torch.zeros((b, f * N), device=q.device, dtype=torch.bool)
        
        last_start = kv_cache.get("last_start", -1)
        if last_start != current_start:
            kv_cache["step_idx"] = 0
            kv_cache["last_start"] = current_start

        step_idx = kv_cache["step_idx"]
        kv_cache["step_idx"] += 1

        is_first_chunk = current_start < frame_seqlen * 2
        warmup_steps = 3 if is_first_chunk else 1

        if timer: timer.start("reuse_decision")
        if f > 1 and step_idx > warmup_steps:
            q_spatial = q.view(b, f, N, C) 
            
            # 1. 极其轻量级的运动判定：同位置相邻帧的 L1 残差 (Motion Proxy)
            l1_diff = torch.abs(q_spatial[:, 1:] - q_spatial[:, :-1]).sum(dim=-1) # [B, f-1, N]
            
            # 初始化累加器
            if "accumulator" not in kv_cache:
                kv_cache["accumulator"] = torch.zeros((b, N), device=q.device)
            
            mask_2d = torch.zeros((b, f, N), device=q.device, dtype=torch.bool)
            alpha_floor = 0.6 # 论文中的背景兜底权重
            tau = 2.0         # 累加器激活阈值 (可调)

            for i in range(1, f):
                current_diff = l1_diff[:, i-1] # [B, N]
                
                # 2. Min-Max 软映射
                diff_min = current_diff.amin(dim=-1, keepdim=True)
                diff_max = current_diff.amax(dim=-1, keepdim=True)
                # 防止除0
                M_norm = (current_diff - diff_min) / (diff_max - diff_min + 1e-6)
                
                # W 代表需要被计算的渴望度 (1=极高频运动，0.6=纯静止背景)
                W = alpha_floor + (1 - alpha_floor) * M_norm 
                
                # 3. 误差累加与门控判定
                kv_cache["accumulator"] += W
                compute_decision = kv_cache["accumulator"] > tau
                
                # 清零已被激活的累加器
                kv_cache["accumulator"][compute_decision] = 0.0
                
                # True 表示跳过计算 (Reuse)
                reuse_decision = ~compute_decision 
                
                # 平滑去碎片的 Block Voting (保持 Attention 的局部连续性)
                block_size = 12
                num_blocks = N // block_size
                mask_blocks = reuse_decision.view(b, num_blocks, block_size)
                # 若 Block 内超过 80% Token 想要休眠，就整体休眠
                block_decision = (mask_blocks.float().mean(dim=-1, keepdim=True) >= 0.8)
                reuse_decision = block_decision.expand(-1, -1, block_size).reshape(b, N)
                
                mask_2d[:, i, :] = reuse_decision
                
            mask_1d = mask_2d.view(b, f * N)

        if injected_reuse_mask is not None:
            mask_1d = injected_reuse_mask
        if timer: timer.end()

        # ---------------------------------------------------------------
        # Attention computation (Sparse or Full)
        # ---------------------------------------------------------------
        # ---------------------------------------------------------------
        # Attention computation (Sparse or Full)
        # ---------------------------------------------------------------
        if timer: timer.start("attention")
        if mask_1d.any() and b == 1:
            # 提取存活的 Token 的索引掩码
            alive_mask = ~mask_1d.view(-1) # [s]
            
            # roped_query: [b, s, n, d] -> 抽取出 [b, num_alive, n, d]
            q_alive = roped_query[:, alive_mask, :, :]
            
            # 仅对存活的 Token 执行 Attention，计算量随存活数量严格线性下降！
            x_alive = attention(q_alive, k_cache_sliced, v_cache_sliced)
            
            # 创建空画布并填回
            x_out_hilbert = torch.zeros_like(roped_query)
            x_out_hilbert[:, alive_mask, :, :] = x_alive
        else:
            x_out_hilbert = attention(roped_query, k_cache_sliced, v_cache_sliced)
        if timer: timer.end()

        # ---------------------------------------------------------------
        # In-place Token Copy (No gather needed!)
        # ---------------------------------------------------------------
        if timer: timer.start("inplace_copy")
        if mask_1d.any():
            x_out_spatial = x_out_hilbert.view(b, f, N, C)
            mask_2d = mask_1d.view(b, f, N)
            for i in range(1, f):
                m = mask_2d[:, i, :].unsqueeze(-1) # [B, N, 1]
                if m.any():
                    # 极其暴力且极速：原地照抄上一帧同一位置的特征
                    x_out_spatial[:, i, :] = torch.where(m, x_out_spatial[:, i-1, :], x_out_spatial[:, i, :])
        if timer: timer.end()

        # ---------------------------------------------------------------
        # Restore to 2D Raster (Only ONCE at the end of the layer)
        # ---------------------------------------------------------------
        if timer: timer.start("restore_raster")
        x_out_hilbert_spatial = x_out_hilbert.view(b, f, N, C)
        x_out_raster = torch.empty_like(x_out_hilbert_spatial)
        # 逆映射：将 Hilbert 排序的数据放回真实的 Raster 坐标中
        x_out_raster[:, :, self.hilbert_to_raster, :] = x_out_hilbert_spatial
        x_out = x_out_raster.view(b, f * N, n, d)
        
        x_out = x_out.flatten(2)
        x_out = self.o(x_out)
        
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        if timer: timer.end()
        
        return x_out


# ---------------------------------------------------------------------------
# Synthetic mask generation
# ---------------------------------------------------------------------------
def generate_reuse_mask(batch_size, num_frames, frame_seqlen, ratio, pattern="block", device="cuda"):
    mask = torch.zeros(batch_size, num_frames * frame_seqlen, dtype=torch.bool, device=device)
    if num_frames <= 1 or ratio <= 0: return mask
    N = frame_seqlen
    reusable_len = (num_frames - 1) * N
    num_reuse = int(reusable_len * ratio)

    if pattern == "block":
        block_size = 12
        total_blocks = reusable_len // block_size
        num_blocks_to_reuse = max(1, int(total_blocks * ratio))
        chosen = torch.randperm(total_blocks, device=device)[:num_blocks_to_reuse].sort().values
        for blk in chosen:
            flat_start = blk.item() * block_size
            frame_idx = flat_start // N + 1
            offset = flat_start % N
            start = frame_idx * N + offset
            mask[:, start : start + block_size] = True
    else:  
        reusable_indices = torch.randperm(reusable_len, device=device)[:num_reuse]
        actual_indices = reusable_indices + N  
        mask[:, actual_indices] = True

    return mask


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def build_kv_cache(batch_size, max_len, n_heads, head_dim, device):
    return {
        "k": torch.randn(batch_size, max_len, n_heads, head_dim, device=device, dtype=torch.bfloat16),
        "v": torch.randn(batch_size, max_len, n_heads, head_dim, device=device, dtype=torch.bfloat16),
        "global_end_index": torch.tensor(0, device=device),
        "local_end_index": torch.tensor(0, device=device),
        "step_idx": torch.tensor(10, device=device),
        "last_start": torch.tensor(-1, device=device),
    }


def benchmark(attn, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, reuse_mask, num_runs, device):
    target_dtype = torch.bfloat16
    x = x.to(dtype=target_dtype)
    freqs = freqs.to(dtype=target_dtype)
    
    timer = CUDATimer()
    with torch.inference_mode():
        # Warm-up
        for _ in range(5):
            _ = attn(x, seq_lens, grid_sizes, freqs, kv_cache, current_start=current_start, injected_reuse_mask=reuse_mask, timer=timer)
            timer.flush()
        timer.reset()

        for _ in range(num_runs):
            kv_cache["global_end_index"].fill_(current_start)
            kv_cache["local_end_index"].fill_(current_start)
            kv_cache["last_start"] = torch.tensor(current_start, device=device)
            kv_cache["step_idx"] = torch.tensor(5, device=device)
            
            # 重置模拟累加器
            if "accumulator" in kv_cache:
                kv_cache["accumulator"].zero_()
        
            out = attn(x, seq_lens, grid_sizes, freqs, kv_cache, current_start=current_start, injected_reuse_mask=reuse_mask, timer=timer)
            timer.flush()

    summary = {}
    for k, v in timer.times.items():
        arr = torch.tensor(v[1:])
        summary[k] = {
            "mean_ms": arr.mean().item(),
            "std_ms": arr.std().item(),
        }
    return summary, out


def main():
    parser = argparse.ArgumentParser(description="Benchmark Self-Forcing Attention Reuse - MotionCache V2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--frame_seqlen", type=int, default=1560)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--local_attn_size", type=int, default=-1)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--reuse_ratios", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95])
    parser.add_argument("--reuse_pattern", type=str, default="block", choices=["random", "block"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: B={args.batch_size}, F={args.num_frames}, N={args.frame_seqlen}, D={args.dim}, H={args.num_heads}")
    print("-" * 80)

    dim, num_heads, b, f, N = args.dim, args.num_heads, args.batch_size, args.num_frames, args.frame_seqlen
    head_dim = dim // num_heads
    total_len = f * N

    attn = BenchmarkCausalWanSelfAttention(dim=dim, num_heads=num_heads, local_attn_size=args.local_attn_size, sink_size=args.sink_size).to(device, dtype=torch.bfloat16).eval()

    x = torch.randn(b, total_len, dim, device=device, dtype=torch.bfloat16)
    seq_lens = torch.full((b,), total_len, dtype=torch.long, device=device)
    grid_sizes = torch.tensor([[f, 30, 52]], dtype=torch.long, device=device)
    freqs = torch.randn(1024, head_dim // 2, device=device, dtype=torch.bfloat16)

    cache_capacity = max(32768, total_len * 4)
    kv_cache = build_kv_cache(b, cache_capacity, num_heads, head_dim, device)

    current_start = 2 * N
    kv_cache["global_end_index"].fill_(current_start)
    kv_cache["local_end_index"].fill_(current_start)
    kv_cache["k"][:, :current_start] = torch.randn_like(kv_cache["k"][:, :current_start])
    kv_cache["v"][:, :current_start] = torch.randn_like(kv_cache["v"][:, :current_start])

    results = []
    for ratio in args.reuse_ratios:
        mask = generate_reuse_mask(b, f, N, ratio, pattern=args.reuse_pattern, device=device) if ratio > 0 else None
        label = f"reuse={ratio:.2f}"
        summary, _ = benchmark(attn, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, mask, args.num_runs, device)
        results.append((label, summary))

    stages_to_print = [
        ("rope_and_hilbert", "Rope & 1D Hilbert Mapping"),
        ("kv_cache_update",  "KV Cache Update"),
        ("reuse_decision",   "L1 Diff & Accumulator Logic"),
        ("attention",        "Attention Kernel"),
        ("inplace_copy",     "In-place Feature Copy"),
        ("restore_raster",   "Restore to 2D Raster"),
    ]

    print("\n" + "="*95)
    header = f"{'Stage':<30} | " + " | ".join([f"{lbl:<15}" for lbl, _ in results])
    print(header)
    print("-" * len(header))
    
    for stage_key, display_name in stages_to_print:
        row = f"{display_name:<30} | "
        for lbl, summary in results:
            if stage_key in summary:
                mean = summary[stage_key]["mean_ms"]
                std = summary[stage_key]["std_ms"]
                row += f"{mean:5.2f}±{std:4.2f} ms  | "
            else:
                row += f"{'N/A':<15} | "
        print(row)

    print("-" * len(header))
    row_total = f"{'Total Forward Time':<30} | "
    for lbl, summary in results:
        total = sum(summary[k]["mean_ms"] for k in [k for k,_ in stages_to_print] if k in summary)
        row_total += f"{total:5.2f} ms         | "
    print(row_total)

if __name__ == "__main__":
    main()