from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
import torch.nn.functional as F

from time_utils import timer

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        layer_idx=10,       # 【新增】
        record_data=False  # 【新增】
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start
            
        # 新增，保留x的记录    
        x_input_view = x.view(b, s, n, d)

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                with timer("attn_time"):
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                with timer("attn_time"):
                    x = flex_attention(
                        query=padded_roped_query.transpose(2, 1),
                        key=padded_roped_key.transpose(2, 1),
                        value=padded_v.transpose(2, 1),
                        block_mask=block_mask
                    )[:, :, :-padded_length].transpose(2, 1)
        else:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
                f = q.shape[1] // frame_seqlen
                
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

                current_end = current_start + roped_query.shape[1]
                sink_tokens = self.sink_size * frame_seqlen
                
                kv_cache_size = kv_cache["k"].shape[1]
                num_new_tokens = roped_query.shape[1]
                
                # --- 保持原有的 KV Cache 滚动更新逻辑 ---
                if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                        num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
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

                # ====================================================
                # [Self-Forcing] 去噪热身机制 (Warm-up) & 运动补偿 + 方差自适应
                # ====================================================
                # 追踪当前 Block 的去噪步数
                last_start = kv_cache.get("last_start", -1)
                if last_start != current_start:
                    kv_cache["step_idx"] = 0
                    kv_cache["last_start"] = current_start
                    
                step_idx = kv_cache["step_idx"]
                kv_cache["step_idx"] += 1

                # 1. 基础空间维度定义
                H = grid_sizes[0][1].item()
                W = grid_sizes[0][2].item()
                C = n * d          # 特征总通道数
                N = frame_seqlen   # 单帧 Token 数量 (H * W)
                
                reuse_mask = torch.zeros((b, f * N), device=q.device, dtype=torch.bool)
                # 用于记录每一帧每个 Token 的最佳偏移索引 (运动矢量)
                best_indices = torch.zeros((b, f, N), dtype=torch.long, device=q.device)

                # 2. 由粗到细 (Coarse-to-Fine) 动态 kernel_size 调整
                if step_idx <= 1:
                    kernel_size = 5  # 5x5 邻域 (25 个候选方向，捕获大运动)
                elif step_idx <= 3:
                    kernel_size = 3  # 3x3 邻域 (9 个候选方向)
                else:
                    kernel_size = 1  # 1x1 邻域 (无偏移搜索，极速比对)

                # 设定热身步数，比如第 1 步不复用，建立高质量基准帧
                is_first_chunk = (current_start < frame_seqlen * 2) 
                warmup_steps = 3 if is_first_chunk else 1

                if f > 1 and step_idx > warmup_steps:
                    # 将 Query 还原为 2D 空间形状: [B, F, C, H, W]
                    q_spatial = q.view(b, f, H, W, C).permute(0, 1, 4, 2, 3)
                    
                    # === 仅修改搜索维度 (通道降维) ===
                    sub_C = max(C // 4, 1) # 取前 1/4 通道用于搜索
                    q_spatial_lite = q_spatial[:, :, :sub_C, :, :]
                    # 先降维，再对子特征进行 L2 归一化
                    q_spatial_lite_norm = F.normalize(q_spatial_lite, p=2, dim=2)
                    # ===========================================
                    
                    # === [保持原有逻辑] 预先生成 Grid 坐标用于后续 grid_sample 提取 ===
                    iy, ix = torch.meshgrid(torch.arange(H, device=q.device), 
                                            torch.arange(W, device=q.device), indexing='ij')

                    for i in range(1, f):
                        q_i_lite = q_spatial_lite_norm[:, i]       # 降维后的当前帧 [B, sub_C, H, W]
                        q_prev_lite = q_spatial_lite_norm[:, i-1]  # 降维后的上一帧 [B, sub_C, H, W]
                        
                        if kernel_size == 1:
                            # --- 1x1 极速路径 ---
                            # 【修改点】：使用降维特征进行点乘相似度计算
                            max_sim = (q_i_lite * q_prev_lite).sum(dim=1).view(b, N)
                            best_idx = torch.zeros((b, N), dtype=torch.long, device=q.device)
                            
                            # 使用未归一化的原始全维度特征 q_spatial 计算方差，反映真实的能量波动
                            q_i_raw = q_spatial[:, i] 
                            local_variance = q_i_raw.view(b, C, N).var(dim=1) 
                        else:
                            # --- 2D 邻域搜索路径 ---
                            num_offsets = kernel_size * kernel_size
                            pad_size = kernel_size // 2

                            # 局部方差计算 (保持原样，依然使用全维度特征保证方差精度)
                            q_i_raw = q_spatial[:, i] 
                            q_i_raw_sq = q_i_raw ** 2
                            E_x = F.avg_pool2d(q_i_raw, kernel_size=kernel_size, stride=1, padding=pad_size)
                            E_x2 = F.avg_pool2d(q_i_raw_sq, kernel_size=kernel_size, stride=1, padding=pad_size)
                            local_variance = (E_x2 - E_x ** 2).clamp(min=0).mean(dim=1).view(b, N) # [B, N]
                            
                            # 修改点：使用降维特征进行平移搜索
                            q_prev_pad = F.pad(q_prev_lite, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0.0)
                            sims_list = []
                            
                            for dy in range(kernel_size):
                                for dx in range(kernel_size):
                                    q_prev_shifted = q_prev_pad[:, :, dy:dy+H, dx:dx+W]
                                    # 用轻量化特征算点乘
                                    sim = (q_i_lite * q_prev_shifted).sum(dim=1) 
                                    sims_list.append(sim.view(b, N))
                                    
                            sims_stack = torch.stack(sims_list, dim=1) 
                            max_sim, best_idx = sims_stack.max(dim=1)
                        
                        # 记录这一帧的运动矢量
                        best_indices[:, i] = best_idx
                        
                        # ====================================================
                        # 核心修复：相对方差基准与更严格的阈值
                        # ====================================================
                        # 计算当前批次和帧的平均空间方差
                        mean_var = local_variance.mean(dim=1, keepdim=True) # [B, 1]
                        
                        # 判定平坦区：方差低于全图平均水平的 20%
                        is_flat_area = local_variance < (mean_var * 0.2)
                        
                        # 抬高余弦相似度门槛 (0.98 / 0.90)
                        mask_1d = (max_sim > 0.98) | ((max_sim > 0.90) & is_flat_area)
                        
                        start_idx = i * N
                        end_idx = (i + 1) * N
                        reuse_mask[:, start_idx:end_idx] = mask_1d
                        
                # ====================================================
                # 动态 Attention 计算与极速补偿应用
                # ====================================================
                if reuse_mask.any():
                    print(f"步数 {step_idx} - Token 丢弃率: {(reuse_mask.float().mean().item() * 100):.2f}%")
                    
                    if b == 1:
                        alive_mask = ~reuse_mask[0] 
                        q_alive = roped_query[:, alive_mask] 
                        
                        with timer("attn_time"):
                            x_alive = attention(q_alive, k_cache_sliced, v_cache_sliced)
                        
                        x = torch.empty_like(roped_query)
                        x[:, alive_mask] = x_alive
                    else:
                        q_sparse = roped_query.clone()
                        q_sparse[reuse_mask] = 0.0 
                        with timer("attn_time"):
                            x = attention(q_sparse, k_cache_sliced, v_cache_sliced)
                    
                    # 自回归运动补偿复用
                    for i in range(1, f):
                        start_idx = i * N
                        end_idx = (i + 1) * N
                        prev_start = (i - 1) * N
                        prev_end = i * N
                        
                        # 获取当前帧的 1D 掩码: [B, N]
                        mask_i = reuse_mask[:, start_idx:end_idx]
                        
                        if not mask_i.any():
                            continue
                            
                        # 取出上一帧的 Attention 输出，展平特征维度以适配空间操作 [B, N, C]
                        x_prev_1d = x[:, prev_start:prev_end].view(b, N, C)
                        
                        if kernel_size == 1:
                            # 1x1 极速补偿，直接获取原位置特征
                            best_x_1d = x_prev_1d 
                        else:
                            # === [保持原有逻辑] 使用 grid_sample 替换 Gather 逻辑 ===
                            # 还原为 2D 空间
                            x_prev_2d = x_prev_1d.permute(0, 2, 1).view(b, C, H, W)
                            
                            best_idx_i = best_indices[:, i].view(b, H, W) # [B, H, W]
                            pad_size = kernel_size // 2
                            
                            # 解码相对偏移量
                            offset_y = (best_idx_i // kernel_size) - pad_size
                            offset_x = (best_idx_i % kernel_size) - pad_size
                            
                            # 计算绝对目标坐标
                            target_y = iy.unsqueeze(0) + offset_y
                            target_x = ix.unsqueeze(0) + offset_x
                            
                            # 归一化坐标并强制转换为 BFloat16/Float16 以防报错
                            grid_y = (target_y / max(H - 1, 1)) * 2.0 - 1.0
                            grid_x = (target_x / max(W - 1, 1)) * 2.0 - 1.0
                            grid = torch.stack((grid_x, grid_y), dim=-1).to(x_prev_2d.dtype)
                            
                            # nearest 插值极速提取特征
                            best_x_2d = F.grid_sample(x_prev_2d, grid, mode='nearest', 
                                                      padding_mode='border', align_corners=True)
                            
                            # 转置回 1D 形状
                            best_x_1d = best_x_2d.view(b, C, N).permute(0, 2, 1) # [B, N, C]
                            # =========================================================
                        
                        # 还原为标准的 Attention 输出形状: [B, N, n, d]
                        best_x_1d = best_x_1d.view(b, N, n, d)
                        
                        # 应用掩码，替换复用特征
                        mask_expanded = mask_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n, d)
                        x[:, start_idx:end_idx] = torch.where(mask_expanded, best_x_1d, x[:, start_idx:end_idx])
                else:
                    with timer("attn_time"):
                        x = attention(roped_query, k_cache_sliced, v_cache_sliced)

                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)
                
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
        #     frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        #     current_start_frame = current_start // frame_seqlen
        #     roped_query = causal_rope_apply(
        #         q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
        #     roped_key = causal_rope_apply(
        #         k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

        #     current_end = current_start + roped_query.shape[1]
        #     sink_tokens = self.sink_size * frame_seqlen
        #     # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
        #     kv_cache_size = kv_cache["k"].shape[1]
        #     num_new_tokens = roped_query.shape[1]
        #     if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
        #             num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
        #         # Calculate the number of new tokens added in this step
        #         # Shift existing cache content left to discard oldest tokens
        #         # Clone the source slice to avoid overlapping memory error
        #         num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
        #         num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
        #         kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
        #             kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        #         kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
        #             kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        #         # Insert the new keys/values at the end
        #         local_end_index = kv_cache["local_end_index"].item() + current_end - \
        #             kv_cache["global_end_index"].item() - num_evicted_tokens
        #         local_start_index = local_end_index - num_new_tokens
        #         kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        #         kv_cache["v"][:, local_start_index:local_end_index] = v
        #     else:
        #         # Assign new keys/values directly up to current_end
        #         local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
        #         local_start_index = local_end_index - num_new_tokens
        #         kv_cache["k"][:, local_start_index:local_end_index] = roped_key
        #         kv_cache["v"][:, local_start_index:local_end_index] = v
        #     x = attention(
        #         roped_query,
        #         kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
        #         kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
        #     )
        #     # q_in = roped_query
        #     # k_in = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
        #     # v_in = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]

        #     # import os
        #     # import uuid
        #     # if os.environ.get("DUMP_ATTN") == "1":
        #     #     # 1. 转换维度: [B, SeqLen, Heads, HeadDim] -> [B, Heads, SeqLen, HeadDim]
        #     #     q_mat = q_in.transpose(1, 2)
        #     #     k_mat = k_in.transpose(1, 2)
        #     #     v_mat = v_in.transpose(1, 2)
                
        #     #     # 2. 手动计算 Attention Map
        #     #     scale = self.head_dim ** -0.5
        #     #     attn_scores = (q_mat @ k_mat.transpose(-2, -1)) * scale
        #     #     attn_map = attn_scores.softmax(dim=-1)
                
        #     #     # 3. 截获并落盘
        #     #     dump_dict = {
        #     #         'q': q_mat.detach().cpu().bfloat16(),
        #     #         'k': k_mat.detach().cpu().bfloat16(),
        #     #         'v': v_mat.detach().cpu().bfloat16(),
        #     #         'attn_map': attn_map.detach().cpu().bfloat16(),
        #     #         'grid_sizes': grid_sizes.detach().cpu()
        #     #     }
        #     #     filename = f"dump_attn_causal_{uuid.uuid4().hex[:6]}.pt"
        #     #     torch.save(dump_dict, filename)
        #     #     print(f"✅ 成功 Dump 了一层 Causal Attention Map: {filename}")
                
        #     #     # 4. 保持前向传播不断
        #     #     x = (attn_map @ v_mat).transpose(1, 2)
        #     # else:
        #     #     # 正常推理时走原生 attention 算子
        #     #     x = attention(q_in, k_in, v_in)
        #     kv_cache["global_end_index"].fill_(current_end)
        #     kv_cache["local_end_index"].fill_(local_end_index)
            
        # if record_data and layer_idx == 28:
        #     import os
        #     os.makedirs("attn_logs_28_b", exist_ok=True)
            
        #     # 定位到最后一个 Batch 和最后一个 Head
        #     log_data = {
        #         "step": current_start,
        #         "x_norm": x_input_view[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "q_norm": q[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "k_norm": k[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "v_norm": v[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "attn_out_norm": x[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #     }
            
        #     if kv_cache is not None:
        #         # 新增对 v 的提取
        #         log_data["kv_cache_k_state"] = kv_cache["k"][-1, :, -1, :].norm(dim=-1).detach().cpu()
        #         log_data["kv_cache_v_state"] = kv_cache["v"][-1, :, -1, :].norm(dim=-1).detach().cpu()
        #         log_data["local_end"] = kv_cache["local_end_index"].item()
            
        #     torch.save(log_data, f"attn_logs_28_b/step_{current_start}.pt")

        # elif record_data and layer_idx == 29:
        #     import os
        #     os.makedirs("attn_logs_29_b", exist_ok=True)
            
        #     log_data = {
        #         "step": current_start,
        #         "x_norm": x_input_view[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "q_norm": q[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "k_norm": k[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "v_norm": v[-1, :, -1, :].norm(dim=-1).detach().cpu(),
        #         "attn_out_norm": x[-1, :, -1, :].norm(dim=-1).detach().cpu(), 
        #     }
            
        #     if kv_cache is not None:
        #         log_data["kv_cache_k_state"] = kv_cache["k"][-1, :, -1, :].norm(dim=-1).detach().cpu()
        #         log_data["kv_cache_v_state"] = kv_cache["v"][-1, :, -1, :].norm(dim=-1).detach().cpu()
        #         log_data["local_end"] = kv_cache["local_end_index"].item()
            
        #     torch.save(log_data, f"attn_logs_29_b/step_{current_start}.pt")
            

        # output
        # x = x.flatten(2)
        # x = self.o(x)
        # return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        layer_idx=10,       # 新增
        record_data=False  #  新增
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            layer_idx=layer_idx, record_data=record_data 
        )

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            with timer("attn_time"):
                cross_out = self.cross_attn(self.norm3(x), context,
                                            context_lens, crossattn_cache=crossattn_cache)
            x = x + cross_out
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_idx": block_index,  # 当前层号
                        "record_data": True        # 是否记录数据
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_idx": block_index,     # 传递当前层号
                        "record_data": True
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
