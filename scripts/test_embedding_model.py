import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

query = (
    QUERY_INSTRUCTION
    + "ATV340发生过电流故障后应该如何检查？"
)

documents = [
    (
        "过电流故障可能由电机参数不正确、"
        "负载过高或机械堵塞引起，应检查电机参数和机械装置。"
    ),
    (
        "图形显示终端支持修改语言、日期和屏幕亮度。"
    ),
]

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

embeddings = model.encode(
    [query, *documents],
    normalize_embeddings=True,
    show_progress_bar=False,
)

similarities = np.dot(
    embeddings[1:],
    embeddings[0],
)

print(f"模型：{MODEL_NAME}")
print(f"向量形状：{embeddings.shape}")
print(f"相关文本相似度：{similarities[0]:.4f}")
print(f"无关文本相似度：{similarities[1]:.4f}")