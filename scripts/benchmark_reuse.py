"""
Benchmark script for CausalWanSelfAttention with controllable token reuse ratio.

This script isolates the Self-Forcing attention module and measures:
1. Baseline (no reuse) attention latency
2. Latency under synthetic reuse masks with different ratios (0.5, 0.8, 0.95)
3. Breakdown of overhead: rope, KV cache update, reuse-mask logic, attention kernel,
   and motion-compensation gather.
"""

import argparse
import math
import time
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Fallback attention: mimics wan.modules.attention.attention
# ---------------------------------------------------------------------------
try:
    from wan.modules.attention import attention
except Exception:
    def attention(q, k, v):
        """
        q, k, v: [B, L, H, D]
        """
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)


# ---------------------------------------------------------------------------
# Helpers copied from causal_model_new.py
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
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x, y = x + s * rx, y + s * ry
            t //= 4
            s *= 2
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
    """
    修改为异步收集模式：start/end只打Event标，不阻塞同步。
    在每次 forward 结束时调用 flush() 统一等待并计算累加时间。
    这允许我们在 for 循环内部精准测量而不会降低 GPU 并发效率。
    """
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
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache,
        current_start=0,
        cache_start=None,
        injected_reuse_mask: Optional[torch.Tensor] = None,
        injected_best_indices: Optional[torch.Tensor] = None,
        timer: Optional[CUDATimer] = None,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # ---------------------------------------------------------------
        # Rope
        # ---------------------------------------------------------------
        if timer: timer.start("rope")
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        f = q.shape[1] // frame_seqlen
        current_start_frame = current_start // frame_seqlen
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        if timer: timer.end()

        # ---------------------------------------------------------------
        # KV cache update
        # ---------------------------------------------------------------
        if timer: timer.start("kv_cache_update")
        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]

        if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
            num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size
        ):
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        k_cache_sliced = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
        v_cache_sliced = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
        if timer: timer.end()

        # ---------------------------------------------------------------
        # Reuse mask logic (internal computation)
        # ---------------------------------------------------------------
        C = n * d
        N = frame_seqlen

        reuse_mask = torch.zeros((b, f * N), device=q.device, dtype=torch.bool)
        best_indices_hilbert = torch.zeros((b, f, N), dtype=torch.long, device=q.device)

        if timer: timer.start("reuse_logic_total")

        last_start = kv_cache.get("last_start", -1)
        if last_start != current_start:
            kv_cache["step_idx"] = 0
            kv_cache["last_start"] = current_start

        step_idx = kv_cache["step_idx"]
        kv_cache["step_idx"] += 1

        if step_idx <= 1:
            search_radius = 13
        elif step_idx <= 3:
            search_radius = 5
        else:
            search_radius = 1

        is_first_chunk = current_start < frame_seqlen * 2
        warmup_steps = 3 if is_first_chunk else 1

        if f > 1 and step_idx > warmup_steps:
            if timer: timer.start("reuse_prep")
            q_spatial = q.view(b, f, N, C)
            q_hilbert = q_spatial[:, :, self.hilbert_to_raster, :] # Hilbert 重排
            sub_C = max(C // 4, 1)
            q_hilbert_lite = q_hilbert[:, :, :, :sub_C].permute(0, 1, 3, 2)
            q_hilbert_lite_norm = F.normalize(q_hilbert_lite, p=2, dim=2)
            q_hilbert_raw = q_hilbert.permute(0, 1, 3, 2)
            if timer: timer.end()

            for i in range(1, f):
                q_i_lite = q_hilbert_lite_norm[:, i]
                q_prev_lite = q_hilbert_lite_norm[:, i - 1]
                q_raw_i = q_hilbert_raw[:, i]

                # 1. 邻域搜索 (Neighborhood Search)
                if timer: timer.start("reuse_search")
                if search_radius == 0:
                    max_sim = (q_i_lite * q_prev_lite).sum(dim=1)
                    best_idx = torch.arange(N, device=q.device).expand(b, N)
                    if timer: timer.end() # 结束 Search

                    # 2. 方差计算 (Variance Calc)
                    if timer: timer.start("reuse_variance")
                    local_variance = q_raw_i.var(dim=1)
                    if timer: timer.end()
                else:
                    # 1. 依然做 Padding
                    # q_prev_lite: [B, sub_C, N]
                    q_prev_pad = F.pad(q_prev_lite, (search_radius, search_radius), mode='constant', value=0.0)
    
                    # 2. 使用 unfold 提取滑动窗口，这一步避免了 for 循环
                    # 形状变为: [B, sub_C, N, 2 * search_radius + 1]
                    window_size = search_radius * 2 + 1
                    q_prev_windows = q_prev_pad.unfold(dimension=-1, size=window_size, step=1)
    
                    # 3. 利用广播机制直接计算相似度，并沿 sub_C (dim=1) 求和
                    # q_i_lite.unsqueeze(-1) 形状: [B, sub_C, N, 1]
                    # sims_stack 形状: [B, N, 2 * search_radius + 1]
                    sims_stack = (q_i_lite.unsqueeze(-1) * q_prev_windows).sum(dim=1)
    
                    # 4. 直接在最后一个维度找最大值
                    max_sim, best_offset_idx = sims_stack.max(dim=-1)
    
                    base_indices = torch.arange(N, device=q.device).expand(b, N)
                    best_idx = (base_indices + best_offset_idx - search_radius).clamp(0, N - 1)
                    if timer: timer.end() # 结束 Search

                    # 2. 方差计算 (Variance Calc)
                    if timer: timer.start("reuse_variance")
                    var_window = max(3, search_radius)
                    pad_var = var_window // 2
                    q_raw_sq = q_raw_i ** 2
                    E_x = F.avg_pool1d(q_raw_i, kernel_size=var_window, stride=1, padding=pad_var)
                    E_x2 = F.avg_pool1d(q_raw_sq, kernel_size=var_window, stride=1, padding=pad_var)
                    local_variance = (E_x2 - E_x ** 2).clamp(min=0).mean(dim=1)
                    if timer: timer.end()

                # 3. 多数表决与决策 (Voting Logic)
                # if timer: timer.start("reuse_voting")
                # best_indices_hilbert[:, i] = best_idx
                # mean_var = local_variance.mean(dim=1, keepdim=True)
                # is_flat_area = local_variance < (mean_var * 0.2)
                # token_mask = (max_sim > 0.98) | ((max_sim > 0.90) & is_flat_area)

                # block_size = 12
                # num_blocks = N // block_size
                # mask_blocks = token_mask.view(b, num_blocks, block_size)
                # threshold_ratio = 0.8
                # block_decision = (mask_blocks.float().mean(dim=-1, keepdim=True) >= threshold_ratio)
                # mask_1d_hilbert = block_decision.expand(-1, -1, block_size).reshape(b, N)
                # if timer: timer.end()
                
                if timer: timer.start("reuse_voting")
                best_indices_hilbert[:, i] = best_idx

                # ----- 引入 MotionCache 的 alpha 软映射机制 -----
                # 1. 在当前帧内对方差进行 Min-Max 归一化 (代表运动显著性)
                var_min = local_variance.amin(dim=-1, keepdim=True)
                var_max = local_variance.amax(dim=-1, keepdim=True)
                M_norm = (local_variance - var_min) / (var_max - var_min + 1e-6) # [B, N], 值域 0~1

                # 2. 应用软映射，alpha 作为底噪 (论文推荐 0.6，代表即使是最静止的区域也有 60% 的基础"运动权重")
                alpha = 0.6
                W = alpha + (1 - alpha) * M_norm  # W 越接近1，说明运动越剧烈，越需要重算

                # 3. 将 W 作为动态阈值，替换生硬的 is_flat_area
                # 如果 token 的最大相似度很高 (>0.90) 且它的运动权重较低，就允许复用
                token_mask = (max_sim > 0.98) | ((max_sim > 0.90) & (W < 0.75)) # 0.75可微调

                block_size = 12
                num_blocks = N // block_size
                mask_blocks = token_mask.view(b, num_blocks, block_size)

                # 动态调整多数表决：运动越剧烈的 Block，需要的同意复用比例越高
                block_W = W.view(b, num_blocks, block_size).mean(dim=-1, keepdim=True) # Block的平均运动权重
                dynamic_threshold = 0.6 + 0.3 * block_W # 运动权重大，阈值最高可达0.9；运动小，阈值降到0.6+

                block_decision = (mask_blocks.float().mean(dim=-1, keepdim=True) >= dynamic_threshold)
                mask_1d_hilbert = block_decision.expand(-1, -1, block_size).reshape(b, N)
                if timer: timer.end()

                # 4. 掩码 Hilbert -> Raster 映射 (Mask Map H->R)
                if timer: timer.start("reuse_mask_map")
                start_idx = i * N
                end_idx = (i + 1) * N
                reuse_mask_raster = torch.zeros((b, N), device=q.device, dtype=torch.bool)
                reuse_mask_raster[:, self.hilbert_to_raster] = mask_1d_hilbert
                reuse_mask[:, start_idx:end_idx] = reuse_mask_raster
                if timer: timer.end()

        if timer: timer.end() # End reuse_logic_total

        # ---------------------------------------------------------------
        # Inject synthetic mask if requested
        # ---------------------------------------------------------------
        if injected_reuse_mask is not None:
            reuse_mask = injected_reuse_mask
            if injected_best_indices is not None:
                best_indices_hilbert = injected_best_indices
            else:
                # identity mapping so gather still runs but semantics stay valid
                best_indices_hilbert = torch.arange(N, device=q.device).expand(b, f, N).clone()

        # ---------------------------------------------------------------
        # Attention computation
        # ---------------------------------------------------------------
        if timer: timer.start("attention")
        if reuse_mask.any():
            if b == 1:
                alive_mask = ~reuse_mask[0]
                q_alive = roped_query[:, alive_mask]
                x_alive = attention(q_alive, k_cache_sliced, v_cache_sliced)
                x_out = torch.zeros_like(roped_query)
                x_out[:, alive_mask] = x_alive
            else:
                q_sparse = roped_query.clone()
                q_sparse[reuse_mask] = 0.0
                x_out = attention(q_sparse, k_cache_sliced, v_cache_sliced)
        else:
            x_out = attention(roped_query, k_cache_sliced, v_cache_sliced)
        if timer: timer.end()

        # ---------------------------------------------------------------
        # Motion compensation (gather from prev frame)
        # ---------------------------------------------------------------
        if timer: timer.start("compensation")
        if reuse_mask.any():
            for i in range(1, f):
                start_idx = i * N
                end_idx = (i + 1) * N
                prev_start = (i - 1) * N
                prev_end = i * N
                mask_i_raster = reuse_mask[:, start_idx:end_idx]
                if not mask_i_raster.any():
                    continue
                x_prev_raster = x_out[:, prev_start:prev_end].view(b, N, C)
                x_prev_hilbert = x_prev_raster[:, self.hilbert_to_raster, :]
                best_idx_h = best_indices_hilbert[:, i]
                expanded_indices = best_idx_h.unsqueeze(-1).expand(-1, -1, C)
                best_x_hilbert = torch.gather(x_prev_hilbert, dim=1, index=expanded_indices)
                best_x_raster = best_x_hilbert[:, self.raster_to_hilbert, :]
                best_x_raster = best_x_raster.view(b, N, n, d)
                mask_expanded = mask_i_raster.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, d)
                x_out[:, start_idx:end_idx] = torch.where(mask_expanded, best_x_raster, x_out[:, start_idx:end_idx])
        if timer: timer.end()

        # ---------------------------------------------------------------
        # Output projection
        # ---------------------------------------------------------------
        x_out = x_out.flatten(2)
        x_out = self.o(x_out)

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        return x_out


# ---------------------------------------------------------------------------
# Synthetic mask generation
# ---------------------------------------------------------------------------
def generate_reuse_mask(
    batch_size: int,
    num_frames: int,
    frame_seqlen: int,
    ratio: float,
    pattern: str = "block",
    device: str = "cuda",
):
    mask = torch.zeros(batch_size, num_frames * frame_seqlen, dtype=torch.bool, device=device)
    if num_frames <= 1 or ratio <= 0:
        return mask

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


def benchmark(
    attn: BenchmarkCausalWanSelfAttention,
    x: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: dict,
    current_start: int,
    reuse_mask: Optional[torch.Tensor],
    num_runs: int,
    device: str,
):
    target_dtype = torch.bfloat16
    x = x.to(dtype=target_dtype)
    freqs = freqs.to(dtype=target_dtype)
    
    timer = CUDATimer()
    with torch.inference_mode():
        # Warm-up
        for _ in range(5):
            _ = attn(
                x, seq_lens, grid_sizes, freqs, kv_cache,
                current_start=current_start,
                injected_reuse_mask=reuse_mask,
                timer=timer,
            )
            timer.flush() # 统一冲洗
        timer.reset()

        for _ in range(num_runs):
            kv_cache["global_end_index"].fill_(current_start)
            kv_cache["local_end_index"].fill_(current_start)
            kv_cache["last_start"] = torch.tensor(current_start, device=device)
            kv_cache["step_idx"] = torch.tensor(5, device=device)
        
            out = attn(
                x, seq_lens, grid_sizes, freqs, kv_cache,
                current_start=current_start,
                injected_reuse_mask=reuse_mask,
                timer=timer,
            )
            timer.flush() # 每次Forward结束后统一Flush并累计时间

    # Summarize
    summary = {}
    for k, v in timer.times.items():
        arr = torch.tensor(v[1:])  # drop first to be safe
        summary[k] = {
            "mean_ms": arr.mean().item(),
            "std_ms": arr.std().item(),
            "min_ms": arr.min().item(),
            "max_ms": arr.max().item(),
        }
    return summary, out


def main():
    parser = argparse.ArgumentParser(description="Benchmark Self-Forcing Attention Reuse")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=5, help="Frames per forward call")
    parser.add_argument("--frame_seqlen", type=int, default=1560)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--local_attn_size", type=int, default=-1)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument(
        "--reuse_ratios", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.95]
    )
    parser.add_argument("--reuse_pattern", type=str, default="block", choices=["random", "block"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip_internal_logic",
        action="store_true",
        help="Skip internal reuse-mask computation and inject synthetic mask directly.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: B={args.batch_size}, F={args.num_frames}, N={args.frame_seqlen}, "
          f"D={args.dim}, H={args.num_heads}")
    print("-" * 80)

    dim = args.dim
    num_heads = args.num_heads
    head_dim = dim // num_heads
    b = args.batch_size
    f = args.num_frames
    N = args.frame_seqlen
    total_len = f * N

    attn = BenchmarkCausalWanSelfAttention(
        dim=dim, num_heads=num_heads,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size,
    ).to(device, dtype=torch.bfloat16).eval()

    # Synthetic input
    x = torch.randn(b, total_len, dim, device=device, dtype=torch.bfloat16)
    seq_lens = torch.full((b,), total_len, dtype=torch.long, device=device)
    grid_sizes = torch.tensor([[f, 30, 52]], dtype=torch.long, device=device)

    # Freqs
    c_half = head_dim // 2
    freqs = torch.randn(1024, c_half, device=device, dtype=torch.bfloat16)

    # KV cache
    cache_capacity = max(32768, total_len * 4)
    kv_cache = build_kv_cache(b, cache_capacity, num_heads, head_dim, device)

    # Pre-fill cache 
    current_start = 2 * N
    kv_cache["global_end_index"].fill_(current_start)
    kv_cache["local_end_index"].fill_(current_start)
    kv_cache["k"][:, :current_start] = torch.randn_like(kv_cache["k"][:, :current_start])
    kv_cache["v"][:, :current_start] = torch.randn_like(kv_cache["v"][:, :current_start])

    results = []
    for ratio in args.reuse_ratios:
        if ratio > 0:
            mask = generate_reuse_mask(
                b, f, N, ratio, pattern=args.reuse_pattern, device=device
            )
        else:
            mask = None

        label = f"reuse={ratio:.2f}"
        summary, _ = benchmark(
            attn, x, seq_lens, grid_sizes, freqs, kv_cache,
            current_start=current_start,
            reuse_mask=mask,
            num_runs=args.num_runs,
            device=device,
        )
        results.append((label, summary))

    # ------------------------------------------------------------------
    # Print results (Tree-structured representation)
    # ------------------------------------------------------------------
    stages_to_print = [
        ("rope", "Rope Apply"),
        ("kv_cache_update", "KV Cache Update"),
        ("reuse_logic_total", "Reuse Logic (Total)"),
        ("reuse_prep", "  ├─ Feature Prep & Hilbert"),
        ("reuse_search", "  ├─ Neighbor Search"),
        ("reuse_variance", "  ├─ Variance Calc"),
        ("reuse_voting", "  ├─ Voting Logic"),
        ("reuse_mask_map", "  └─ Mask Map (H->R)"),
        ("attention", "Attention Kernel"),
        ("compensation", "Motion Compensation"),
    ]

    print("\n" + "="*85)
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
        # 只累加最外层的总模块，避免包含子模块(reuse_prep等)的双重计算
        total = sum(summary[k]["mean_ms"] for k in ["rope", "kv_cache_update", "reuse_logic_total", "attention", "compensation"] if k in summary)
        row_total += f"{total:5.2f} ms         | "
    print(row_total)

    # Extra: compute achieved token reduction in attention
    print("\n" + "-" * 80)
    print("Attention token reduction analysis (relative to full cache):")
    for ratio in args.reuse_ratios:
        if ratio > 0:
            mask = generate_reuse_mask(b, f, N, ratio, pattern=args.reuse_pattern, device=device)
            num_alive = (~mask).sum().item()
        else:
            num_alive = total_len
        reduction = 1.0 - (num_alive / total_len)
        print(f"  Reuse ratio {ratio:.2f}: alive tokens = {num_alive}/{total_len} "
              f"(reduction {reduction*100:.1f}%)")

if __name__ == "__main__":
    main()