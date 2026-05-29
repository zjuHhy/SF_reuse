import torch
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_attention_logs(log_dir="attn_logs_29_b"): 
    # 读取所有保存的步骤数据，并按 step 数字排序
    files = sorted(glob.glob(f"{log_dir}/step_*.pt"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not files:
        print(f"未找到日志文件，请确认 {log_dir} 文件夹存在且有数据。")
        return
    
    # 动态提取 log_dir 后缀，防止图片互相覆盖
    suffix = log_dir.split('logs_')[-1] if 'logs_' in log_dir else "custom"
    
    os.makedirs("table_total", exist_ok=True)
    all_x, all_q, all_k, all_v, all_attn = [], [], [], [], []
    k_cache_states = []
    v_cache_states = [] 
    
    print(f"正在分析 {len(files)} 个推理步骤的数据...")

    for f in files:
        data = torch.load(f, weights_only=True)
        # 将张量展平为一维数组，方便画散点图
        all_x.extend(data["x_norm"].float().numpy().flatten())
        all_q.extend(data["q_norm"].float().numpy().flatten())
        all_k.extend(data["k_norm"].float().numpy().flatten())
        all_v.extend(data["v_norm"].float().numpy().flatten())
        all_attn.extend(data["attn_out_norm"].float().numpy().flatten())
        
        # 收集每一步的 K 状态
        if "kv_cache_k_state" in data:
            k_cache_states.append(data["kv_cache_k_state"].float().numpy())
            
        # 收集每一步的 V 状态
        if "kv_cache_v_state" in data:
            v_cache_states.append(data["kv_cache_v_state"].float().numpy())

    # ==========================================
    # 图 1：X, Q, K, V 的幅度 vs Attention 输出幅度
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle(f"Relationship between X, Q, K, V and Attention Output (L2 Norms) - {suffix}", fontsize=16)

    sns.scatterplot(x=all_x, y=all_attn, ax=axes[0], alpha=0.3, color='blue', edgecolor=None)
    axes[0].set_xlabel("Input X Norm")
    axes[0].set_ylabel("Attention Output Norm")

    sns.scatterplot(x=all_q, y=all_attn, ax=axes[1], alpha=0.3, color='red', edgecolor=None)
    axes[1].set_xlabel("Query Q Norm")

    sns.scatterplot(x=all_k, y=all_attn, ax=axes[2], alpha=0.3, color='green', edgecolor=None)
    axes[2].set_xlabel("Key K Norm")

    sns.scatterplot(x=all_v, y=all_attn, ax=axes[3], alpha=0.3, color='purple', edgecolor=None)
    axes[3].set_xlabel("Value V Norm")

    plt.tight_layout()
    plt.savefig(f"table_total/qkv_vs_attention_{suffix}.png", dpi=150)
    print(f"已保存相关性散点图: qkv_vs_attention_{suffix}.png")

    # ==========================================
    # 图 2：K Cache Recache/Rolling 热力图
    # ==========================================
    if k_cache_states:
        max_len_k = max(len(c) for c in k_cache_states)
        heatmap_data_k = np.zeros((len(k_cache_states), max_len_k))
        
        for i, c in enumerate(k_cache_states):
            heatmap_data_k[i, :len(c)] = c
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data_k, cmap="viridis", cbar_kws={'label': 'Key Token L2 Norm'})
        plt.title(f"K Cache Recache progression Over Generation Steps - {suffix}")
        plt.xlabel("Token Position in K Cache Buffer")
        plt.ylabel("Generation Step")
        
        plt.tight_layout()
        plt.savefig(f"table_total/k_cache_progression_{suffix}.png", dpi=150)
        print(f"已保存 K Cache 热力图: k_cache_progression_{suffix}.png")

    # ==========================================
    # 图 3：V Cache Recache/Rolling 热力图
    # ==========================================
    if v_cache_states:
        max_len_v = max(len(c) for c in v_cache_states)
        heatmap_data_v = np.zeros((len(v_cache_states), max_len_v))
        
        for i, c in enumerate(v_cache_states):
            heatmap_data_v[i, :len(c)] = c
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data_v, cmap="plasma", cbar_kws={'label': 'Value Token L2 Norm'})
        plt.title(f"V Cache Recache progression Over Generation Steps - {suffix}")
        plt.xlabel("Token Position in V Cache Buffer")
        plt.ylabel("Generation Step")
        
        plt.tight_layout()
        plt.savefig(f"table_total/v_cache_progression_{suffix}.png", dpi=150)
        print(f"已保存 V Cache 热力图: v_cache_progression_{suffix}.png")

if __name__ == "__main__":
    analyze_attention_logs("attn_logs_29_b")
    # analyze_attention_logs("attn_logs_29_b")