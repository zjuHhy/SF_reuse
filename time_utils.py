import contextlib
import time
import torch

# 公共黑板：用于存储跨文件统计的时间数据
profiling_stats = {
    "attn_time": 0.0
}

@contextlib.contextmanager
def timer(key):
    """用于精准测量 GPU 算子耗时的上下文管理器"""
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    profiling_stats[key] += time.time() - start