运行结果
train_loss=0.5185 dev_loss=0.3762 AUC=0.6515 MRR=0.3560 nDCG@5=0.3381 nDCG@10=0.4016。

模型输入:batch 是一个dict，共八个字段
```
batch = move_batch_to_device(batch, device)
```

![[Pasted image 20260404113152.png|181]]![[Pasted image 20260404113227.png|377]]
进入model
```
outputs = model(
	history_ids=batch["history_ids"],
	history_mask=batch["history_mask"],
	candidate_ids=batch["candidate_ids"],
	candidate_mask=batch["candidate_mask"],
)
```
进入forward
```
def forward(
	self,
	history_ids: torch.Tensor,
	history_mask: torch.Tensor,
	candidate_ids: torch.Tensor,
	candidate_mask: torch.Tensor,
	) -> dict[str, torch.Tensor]:
	history_news_vecs = self.encode_news_batch(history_ids)
	candidate_news_vecs = self.encode_news_batch(candidate_ids)
	user_vec = self.user_encoder(history_news_vecs, history_mask)
	logits = self.scorer(user_vec, candidate_news_vecs)
	logits = logits.masked_fill(~candidate_mask, -1e9)
	return {
	"logits": logits,
	"user_vec": user_vec,
	"history_news_vecs": history_news_vecs,
	"candidate_news_vecs": candidate_news_vecs,
	}
```
先进行新闻history embedding
```
history_news_vecs = self.encode_news_batch(history_ids)


步入
def encode_news_batch(self, news_ids: torch.Tensor) -> torch.Tensor:
features = self.lookup_news_features(news_ids)
return self.news_encoder(
	category_ids=features["category_ids"],
	subcategory_ids=features["subcategory_ids"],
	title_token_ids=features["title_token_ids"],
	title_token_mask=features["title_token_mask"],
	entity_ids=features.get("entity_ids"),
	entity_mask=features.get("entity_mask"),
)

步入
def lookup_news_features(self, news_ids: torch.Tensor) -> dict[str, torch.Tensor]:
	features = {
	"category_ids": self.news_category_ids[news_ids],
	"subcategory_ids": self.news_subcategory_ids[news_ids],
	"title_token_ids": self.news_title_token_ids[news_ids],
	"title_token_mask": self.news_title_mask[news_ids],
	}
	if self.use_entities:
	features["entity_ids"] = self.news_entity_ids[news_ids]
	features["entity_mask"] = self.news_entity_mask[news_ids]
	return features
```
1个batch里8个样本（用户），41个历史行为（41个被点击的新闻）
![[Pasted image 20260404120723.png|235]]
![[Pasted image 20260404120806.png|463]]
```
步入
return self.news_encoder(
	category_ids=features["category_ids"],
	subcategory_ids=features["subcategory_ids"],
	title_token_ids=features["title_token_ids"],
	title_token_mask=features["title_token_mask"],
	entity_ids=features.get("entity_ids"),
	entity_mask=features.get("entity_mask"),
)
```
进入之后
```
def forward(
    self,
    category_ids: torch.Tensor,
    subcategory_ids: torch.Tensor,
    title_token_ids: torch.Tensor,
    title_token_mask: torch.Tensor,
    entity_ids: Optional[torch.Tensor] = None,
    entity_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # category_ids.shape: [B, L]，例如 [8, 41].把原始二维形状记下来，后面还原用.
    original_shape = category_ids.shape  # original_shape = torch.Size([B, L])

    # category_ids 里一共有多少个元素.[B, L] -> 标量 B*L， 8*41 = 328.
    flat_size = category_ids.numel()  # flat_size = B * L

    # 把 category_ids 从二维摊平成一维. [B, L] -> [B*L]，例如 [8, 41] -> [328]
    flat_category_ids = category_ids.reshape(-1)

    # 把 subcategory_ids 从二维摊平成一维 . [B, L] -> [B*L]，例如 [8, 41] -> [328]
    flat_subcategory_ids = subcategory_ids.reshape(-1)

    # 把title_token_ids的前两维合并，保留最后一维. [8, 41, 24] -> [328, 24]
    flat_title_token_ids = title_token_ids.reshape(flat_size, title_token_ids.size(-1))

    # title_token_mask 同样把前两维合并. [8, 41, 24] -> [328, 24]
    flat_title_token_mask = title_token_mask.reshape(flat_size, title_token_mask.size(-1))

    # 用类别 id 去查 embedding 表
    # 输入: [B*L]. 输出: [B*L, category_dim]， [328] -> [328, 32]
    category_vec = self.category_embedding(flat_category_ids)

    # 用子类别 id 去查 embedding 表
    # 输入: [B*L]. 输出: [B*L, subcategory_dim]，例如 [328] -> [328, 32]
    subcategory_vec = self.subcategory_embedding(flat_subcategory_ids)

    # 用标题 token id 去查 token embedding 表
    # 输入: [B*L, T] .  输出: [B*L, T, token_dim]，例如 [328, 24] -> [328, 24, 128]
    title_token_vecs = self.token_embedding(flat_title_token_ids)

    # 对每条新闻的标题 token 向量做 masked mean pooling
    # 输入:title_token_vecs: [B*L, T, token_dim]  flat_title_token_mask: [B*L, T]
    # 输出:title_vec: [B*L, token_dim]， [328, 24, 128] -> [328, 128]
    title_vec = masked_mean_pool(title_token_vecs, flat_title_token_mask)

    # 把三种特征向量先放进列表里
    # category_vec:[B*L, 32]   subcategory_vec: [B*L, 32]   title_vec:[B*L, 128]
    features = [category_vec, subcategory_vec, title_vec]

    # 如果启用了实体特征，才会走这个分支
    if self.use_entities:
        if entity_ids is None or entity_mask is None:
            raise ValueError("entity_ids and entity_mask are required when use_entities=True")

        # entity_ids 假设原本是 [B, L, E]。合并前两维后变成 [B*L, E]
        flat_entity_ids = entity_ids.reshape(flat_size, entity_ids.size(-1))

        # entity_mask 同理 [B, L, E] -> [B*L, E]
        flat_entity_mask = entity_mask.reshape(flat_size, entity_mask.size(-1))

        # 实体 id 查 embedding 表  。 [B*L, E] -> [B*L, E, entity_dim]
        entity_vecs = self.entity_embedding(flat_entity_ids)

        # 对实体向量做 pooling。 [B*L, E, entity_dim] -> [B*L, entity_dim]
        entity_vec = masked_mean_pool(entity_vecs, flat_entity_mask)

        # 把 entity_vec 也追加到特征列表里  。  当前列表里就会有 4 个 tensor。                  features.append(entity_vec)

    # 沿最后一维拼接所有特征
    # 当前不使用 entity 时：[B*L, 32] + [B*L, 32] + [B*L, 128]   -> [B*L, 192]
    news_input = torch.cat(features, dim=-1)

    # 通过投影层 self.proj:
    # Linear(input_dim -> embedding_dim) + LayerNorm + ReLU + Dropout
    # 维度变化只看输入输出：[B*L, 192] -> [B*L, 128]
    news_vec = self.proj(news_input)

    # 把扁平化后的新闻向量还原回原来的批次结构。[B*L, 128] -> [B, L, 128]
    return news_vec.view(*original_shape, -1)

```
	```
	进入masked_mean_pool
	def masked_mean_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	    # 把布尔 mask 转成 float，方便后面参与乘法和求和，[B, L] -> [B, L]
	    mask = mask.float()
	
	    # 先把 mask 在最后加一维，方便和 sequence 按元素相乘
	    # mask.unsqueeze(-1): [B, L] -> [B, L, 1]
	    # 再和 sequence 相乘，把 padding 位置全部置 0。sequence: [B, L, D]。
	    # 广播后相乘结果:[B, L, D]
	    masked_sequence = sequence * mask.unsqueeze(-1)
	
	    # 对长度维 L 求和，统计每一条样本里有多少个有效位置
	    # mask.sum(dim=1, keepdim=True): [B, L] -> [B, 1]
	    # clamp_min(1.0) 的作用是防止分母变成 0
	    # 如果某一行全是 padding，本来和会是 0，这里强行至少变成 1
	    # 所以最终 denom.shape 还是 [B, 1]
	    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
	
	    # 对 masked_sequence 在长度维 L 上求和[B, L, D] -> [B, D]
	    # 再除以 denom[B, D] / [B, 1] -> 广播后结果 [B, D]
	    # 对每条样本，在有效位置上做平均池化
	    return masked_sequence.sum(dim=1) / denom
	、、、
再进行candidate embedding，candidate_ids 的实际含义是：这一条推荐请求里，每个候选新闻的整数编号。
```
candidate_news_vecs = self.encode_news_batch(candidate_ids)
#[8, 7][B,L]-> [B, L, D][8, 7, 128]
```
再进行把用户的历史点击新闻，压缩成一个用户向量 user_vec。这里L是50，和恰面不一样，理解意义就行。
```
user_vec = self.user_encoder(history_news_vecs, history_mask)
```
![[Pasted image 20260404145444.png|233]]
步入，计算masked_mean_pool(history_news_vecs, history_mask)，统计有效位置的平均值
```
def forward(self, history_news_vecs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor: 
	return masked_mean_pool(history_news_vecs, history_mask)
```
步出，然后根据用户向量 user_vec 和每个候选新闻向量 candidate_news_vecs，算出每个候选新闻的点击分数。
```
logits = self.scorer(user_vec, candidate_news_vecs)
```
![[Pasted image 20260404152132.png|363]]
步入：
```
def forward(self, user_vec: torch.Tensor, candidate_news_vecs: torch.Tensor) -> torch.Tensor:

    # user_vec 原本是每个用户一个向量
    # [B, D], -> [B, 1, D] -> [B, K, D]
    user_vec = user_vec.unsqueeze(1).expand_as(candidate_news_vecs)

    # 按元素相乘，表示用户和候选新闻在每个维度上的交互
    # [B, K, D] * [B, K, D] -> [B, K, D]
    mul_feat = user_vec * candidate_news_vecs

    # 按元素求绝对差，表示用户和候选新闻在每个维度上的距离
    # [B, K, D] - [B, K, D] -> [B, K, D],再取绝对值，结果还是 [B, K, D]
    abs_diff_feat = torch.abs(user_vec - candidate_news_vecs)

    # 对 mul_feat 在最后一维 D 上求和，[B, K, D] -> [B, K, 1]
    dot_feat = mul_feat.sum(dim=-1, keepdim=True)

    # 把多种交互特征在最后一维拼接起来
    #user_vec:[B, K, D]
    # candidate_news_vecs: [B, K, D]
    # mul_feat:            [B, K, D]
    # abs_diff_feat:       [B, K, D]
    # dot_feat:            [B, K, 1]
    # 沿 dim=-1 拼接后：[B, K, D + D + D + D + 1] = [B, K, 4D + 1]
    cross_features = torch.cat(
        [user_vec, candidate_news_vecs, mul_feat, abs_diff_feat, dot_feat],
        dim=-1,
    )

    # 把每个 candidate 的交互特征送入一个 MLP，输出一个标量分数
    # 输入: cross_features: [B, K, 4D + 1]
    # self.mlp 的最后一层输出 1 维，所以先得到：[B, K, 1],
    # squeeze(-1) 把最后那个长度为 1 的维度去掉：[B, K, 1] -> [B, K]
    logits = self.mlp(cross_features).squeeze(-1)

    # 返回每个用户对每个 candidate 的打分
    return logits
```
步出，然后把padding的位置，给予小分,加返回向量
```
logits = logits.masked_fill(~candidate_mask, -1e9)
return {
"logits": logits,
"user_vec": user_vec,
"history_news_vecs": history_news_vecs,
"candidate_news_vecs": candidate_news_vecs,
}
```

步出之后计算loss
```
loss = criterion(
                    outputs["logits"],
                    batch["labels"],
                    batch["candidate_mask"],
                )
```
步入损失函数point loss
```class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = candidate_mask.bool()

        valid_logits = logits[valid_mask]
        valid_labels = labels[valid_mask].float()

        if valid_logits.numel() == 0:
            raise ValueError("No valid candidates found for BCE loss.")

        return F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_labels,
            reduction=self.reduction,
        )

```

## 进入测试阶段
```
dev_metrics = evaluate(model, dev_loader, criterion, device)
```
步入evaluate
```
# 这里取出的 logits 是：[B, K]，第 i 条样本、第 j 个 candidate 的预测分数
logits = outputs["logits"]

# 用当前 batch 的预测分数、真实标签和 candidate mask 计算 loss
# 输入维度：
#   logits:                 [B, K]
#   batch["labels"]:        [B, K]
#   batch["candidate_mask"]:[B, K]
# 这是“当前这个 dev batch 的平均损失”
loss = criterion(logits, batch["labels"], batch["candidate_mask"])

# 统计当前 batch 中真实 candidate 的总数
# batch["candidate_mask"]: [B, K]
# sum() 后把所有 True 加起来：[B, K] -> scalar
valid_count = int(batch["candidate_mask"].sum().item())

# 把当前 batch 的损失累计到 total_loss 里
#loss.item() 通常是“当前 batch 的平均 loss”，所以这里先乘回 valid_count，变成“当前 batch 的总 loss”
#total_loss 表示“从 evaluate 开始到当前 batch 为止，累计的总损失”
total_loss += loss.item() * valid_count

# 把当前 batch 的有效 candidate 数累计起来
total_valid_candidates += valid_count

# 取当前batch的batch size，当前这个batch里有多少条样本impression
# 接下来会按样本（impression）逐条去算 AUC / MRR / nDCG
batch_size = logits.size(0)
# 这里开始按样本逐条计算排序指标
#
# 前面已经有：
#   logits.shape = [B, K]
#   batch["labels"].shape = [B, K]
#   batch["candidate_mask"].shape = [B, K]
#
# 其中：
#   B = 当前 batch 里有多少条 impression / request
#   K = 当前 batch 里 candidate 位置补齐后的最大长度
#
# 时间含义：
#   前面已经完成了“当前 batch 的前向和 loss 计算”
#   从这里开始，是“当前 batch 的指标统计阶段”

for i in range(batch_size):#正在处理“当前 batch 的第 i 条 impression”

    # 取出第 i 条样本的 candidate mask[B, K] -> [K]
    # mask_i[j]表示第 i 条样本第 j 个 candidate 位置,取出第 i 条样本的预测分数，并且只保留真实 candidate
    mask_i = batch["candidate_mask"][i]


    # 当前这条 impression 中，每个真实 candidate 的预测分数
    scores_i = logits[i][mask_i].detach().cpu().tolist()

    # 取出第 i 条样本的真实标签，并且只保留真实 candidate
    #   当前这条 impression 中，每个真实 candidate 的真实标签
    #   一般是 1 表示点击，0 表示未点击
    labels_i = batch["labels"][i][mask_i].detach().cpu().tolist()
    labels_i = [int(x) for x in labels_i]

    # 记录当前已经处理了多少条 request / impression
    request_count += 1

    # 计算当前这条 impression 的 AUC
    #   scores_i: [M]，当前 impression 所有真实 candidate 的分数
    #   labels_i: [M]，对应标签
    auc_i = auc_score(scores_i, labels_i)

   
    # auc_sum: 累计的 AUC 总和
    # auc_count: 成功计算出 AUC 的 request 数量
    if auc_i is not None:
        auc_sum += auc_i
        auc_count += 1

    # 计算当前这条 impression 的 MRR，并累加
    mrr_sum += mrr_score(scores_i, labels_i)

    # 计算当前这条 impression 的 nDCG@5，并累加
    
    ndcg5_sum += ndcg_at_k(scores_i, labels_i, k=5)

    # 计算当前这条 impression 的 nDCG@10，并累加
    
    ndcg10_sum += ndcg_at_k(scores_i, labels_i, k=10)

# evaluate 跑完整个 dev_loader 以后，开始返回整轮验证集的汇总指标
return {

    # 整个验证集的平均 loss
    "loss": total_loss / max(total_valid_candidates, 1),

    # 整个验证集的平均 AUC
    "AUC": auc_sum / max(auc_count, 1),

    # 整个验证集的平均 MRR
    "MRR": mrr_sum / max(request_count, 1),

    # 整个验证集的平均 nDCG@5
    "nDCG@5": ndcg5_sum / max(request_count, 1),

    # 整个验证集的平均 nDCG@10
    "nDCG@10": ndcg10_sum / max(request_count, 1),
}
```
## baseline
- **loss 可以摊平**，因为它是 pointwise 的二分类
- **MRR/nDCG 不能摊平**，因为它们是 listwise / ranking 指标，必须在同一个 impression 内比较
## 接下来是所有指标的计算过程
AUC
```
def auc_score(scores: list[float], labels: list[int]) -> Optional[float]:

    
    #“被点击的 candidate”的预测分数
    pos_scores = [score for score, label in zip(scores, labels) if label == 1]
    # “未点击的 candidate”的预测分数
    neg_scores = [score for score, label in zip(scores, labels) if label == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None

    # correct 用来累计“比较正确”的次数
    correct = 0.0

    # total 用来累计一共比较了多少个“正负样本对”
    # 如果有 P 个正样本，N 个负样本，total = P * N
    total = 0

    # 外层遍历每一个正样本分数
    for pos_score in pos_scores:
        # 内层遍历每一个负样本分数,当前正在把“这个正样本”与所有负样本一一比较
        for neg_score in neg_scores:
            # 如果正样本分数 > 负样本分数,说明模型排序正确,记 1 次正确
            if pos_score > neg_score:
                correct += 1.0
                
            # 如果正样本分数 == 负样本分数,说明模型没有区分开，算半次正确
            elif pos_score == neg_score:
                correct += 0.5

            # 总比较次数 +1
            total += 1

    # 返回 AUC
    
    # 在所有“正样本 vs 负样本”的配对里，有多大比例是正样本分数更高
    return correct / total

```
MRR
```
def mrr_score(scores: list[float], labels: list[int]) -> float:
    # 先把scores和labels一一配对
    # zip(scores, labels) 得到的每个元素都是：(score, label)
    # scores = [0.8,0.3,0.6].  labels = [0,1,0]   [(0.8,0), (0.3,1), (0.6,0)]
    # 当前 impression 中的 candidate，按照模型预测分数从高到低排好序
    
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    
    # 枚举排序后的 candidate，enumerate(..., start=1) 表示排名从1开始
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            return 1.0 / rank #正样本排得越靠前，MRR 越大
    return 0.0

```
ncdg
```
def ndcg_at_k(scores: list[float], labels: list[int], k: int) -> float:
    #scores=[0.8,0.3,0.6]   labels=[0,1,0]   zip后：[(0.8,0),(0.3,1),(0.6,0)]
    # 按score降序排：[(0.8,0),(0.6,0),(0.3,1)]
    # 取 label 后：ranked_labels = [0, 0, 1]
    ranked_labels = [
        label for _, label in sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    ]

    # 构造“理想排序”对应的标签顺序
    ideal_labels = sorted(labels, reverse=True)

    # 计算当前模型排序下的 DCG@k
    dcg = dcg_at_k(ranked_labels, k)

    # 计算理想排序下的 DCG@k
    idcg = dcg_at_k(ideal_labels, k)

    # 如果 idcg == 0.0，说明这条impression的label全是0
    if idcg == 0.0:
        return 0.0

    # 返回归一化后的 nDCG@k . 当前模型排序效果，相对于“理想排序效果”的比例
    return dcg / idcg

```
步入dcg
```
def dcg_at_k(labels: list[int], k: int) -> float:

    # 初始化累计 DCG 分数，当前排序在前 k 个位置上的总收益
    for i, label in enumerate(labels[:k]):
        # 计算“当前位置的收益贡献”
        # 分子部分：(2**label - 1)
        # 如果 label = 1 -> 2**1 - 1 = 1，正样本贡献 1
        # 如果 label = 0 -> 2**0 - 1 = 0，负样本贡献 0
        # 分母部分：math.log2(i + 2)
        # 含义：正样本如果排在前面，收益更大；排得越后，虽然还有收益，但会被打折
        dcg += (2**label - 1) / math.log2(i + 2)
    # 当前这个标签排序在前 k 位上的质量分数
    return dcg
```
