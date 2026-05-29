import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 指向你刚刚拿到的宝贝文件
pt_path = "/home/v-huanghaoyu/projects/dump_attention/dump_attn_causal_0aa308.pt"
print(f"📦 正在加载数据: {pt_path}")
data = torch.load(pt_path, map_location="cpu")

# 2. 提取张量
q = data['q'].float()             # [Batch, Heads, SeqLen, HeadDim]
attn = data['attn_map'].float()   # [Batch, Heads, SeqLen, KV_SeqLen]
grid_sizes = data['grid_sizes'][0].tolist()

print(f"\n✅ 数据加载成功！下面是模型的底层物理结构：")
print(f"📊 Q 向量维度: {q.shape} (Batch, Heads, Query Token数, 隐藏层维度)")
print(f"📊 Attention Map 维度: {attn.shape} (Batch, Heads, Query数, Key/Value数)")
print(f"🎬 视频 Patch 结构: {grid_sizes} (帧数, 高度Patch数, 宽度Patch数)\n")

# ==========================================
# 3. 绘制因果注意力分布 (Causal Attention Matrix)
# ==========================================
plt.figure(figsize=(8, 6))
# 我们取第 0 个 Batch，第 0 个 Head。
# 如果序列很长（几千个 Token），画全图会糊成一团，所以我们截取前 256x256 个 Token 来观察局部的注意力流动
plot_size = min(attn.shape[2], 256)
attn_subset = attn[0, 0, :plot_size, :plot_size].numpy()

# 画图：数值越大颜色越亮
sns.heatmap(attn_subset, cmap="viridis", vmin=0, vmax=np.percentile(attn_subset, 95))
plt.title("Causal Attention Map (Head 0, Subset)")
plt.xlabel("Key/Value Tokens (Past & Present)")
plt.ylabel("Query Tokens (Present)")
plt.tight_layout()
plt.savefig("attention_matrix.png", dpi=300)
print("🖼️ 注意力分布热力图已保存 -> attention_matrix.png")

# ==========================================
# 4. 绘制 Query 向量的自我相似性 (Spatiotemporal Similarity)
# ==========================================
# 取出同一个 Head 的所有 Query
q_head = q[0, 0] # [SeqLen, HeadDim]

# 计算所有 Token 两两之间的余弦相似度 (用来分析时空冗余)
q_norm = F.normalize(q_head, p=2, dim=-1)
sim_matrix = torch.mm(q_norm, q_norm.transpose(0, 1))

plt.figure(figsize=(8, 6))
sim_subset = sim_matrix[:plot_size, :plot_size].numpy()
sns.heatmap(sim_subset, cmap="magma", vmin=0.5, vmax=1.0)
plt.title("Query Token Cosine Similarity (Head 0)")
plt.tight_layout()
plt.savefig("query_similarity.png", dpi=300)
print("🖼️ 时空相似性热力图已保存 -> query_similarity.png")