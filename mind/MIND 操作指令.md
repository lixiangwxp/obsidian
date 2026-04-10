##  生成数据
```
python src/preprocess.py \
  --train-dir data/raw/MINDsmall_train \
  --dev-dir data/raw/MINDsmall_dev \
  --output-dir data/processed \
  --max-history-len 50 \
  --max-title-len 24 \
  --max-entity-len 5 \
  --title-only \
  --train-negative-sample-size 4

```
这条命令的意思是：

- 历史点击最多保留最近 50 条
- 标题 token 最多保留 24 个.每条新闻标题最多保留 24 个 token。减少无效 padding，训练更快。
- 实体最多保留 5 个
- 只用 title，不用 abstract token。先只用标题，不用摘要。
- 训练集每个 impression 保留全部正样本，再随机采样最多 4 个负样本
- dev 不采样，保留完整候选列表，方便后面验证排序效果。验证集不人为缩短候选集合。
## 生成的数据文件
```
Saved news features to data/processed/news_dict.parquet
Saved train impressions to data/processed/train_impressions.parquet
Saved dev impressions to data/processed/dev_impressions.parquet
Saved news id mapping to data/processed/news_id_to_index.json
Saved preprocess metadata to data/processed/preprocess_meta.json
News count: 65238
Train impressions: 156965
Dev impressions: 73152
```
生成数据示例
- news_id_to_index.json：给模型看的“编号字典”
- preprocess_meta.json：给你自己看的“实验记录”
-![[Pasted image 20260403114607.png]]
![[Pasted image 20260403114722.png]]
train_impression
![[Pasted image 20260403114827.png]]
dev_impression
![[Pasted image 20260403115149.png]]
## 生成张量
- 读取 news_dict.parquet
- 读取 news_id_to_index.json
- 构建 baseline 需要的 4 个新闻特征张量：
	- news_category_ids: [65239]：
	- 第 i 个位置存“第 i 条新闻属于哪个 category”； 比如 sports -> 3, health -> 7
    - news_subcategory_ids: [65239]
    -  第 i 个位置存“第 i 条新闻属于哪个 subcategory”；比如 golf -> 12, weightloss -> 18
    - news_title_token_ids: [65239, 24]
    - 第 i 行是一条新闻标题的 token 编号序列，长度固定 24，不够就补 0
    - news_title_mask: [65239, 24]
    - 和 news_title_token_ids 配套，哪些位置是真 token，哪些是 padding
    ![[Pasted image 20260403140217.png|487]]
- 同时返回 3 个词表：
    - category_to_index
    - subcategory_to_index
    - token_to_index
    ![[Pasted image 20260403140655.png|498]]
    某一条新闻编号对应的张量内容
    ![[Pasted image 20260403140940.png]]
   checkpoint 先用 dev MRR 作为“是否保存最好模型”的标准，这对推荐排序任务比较常见。
## 运行指令
conda activate mind
cd /Users/lixiang/Desktop/MIND
PYTHONPATH=src python src/train.py --loss-type pairwise --wandb-project mind --wandb-run-name ablation-pairwise


你这个项目如果加早停，最合适的不是“监控 dev_loss”，而是：

- 用 dev_MRR
- 按 epoch 检查
- 至少跑 3 个 epoch
- 连续 2 个 epoch 没有超过 best 0.001 就停，**当前 epoch 的 dev_MRR 要比历史最佳 best_dev_MRR 至少高一个固定阈值 min_delta。**
- 最终使用 best checkpoint


负采样
- 训练集：
    - 每个 impression 保留**全部正样本**
    - 再从负样本里随机采最多 4 个
- 验证集：
    - 不采样
    - 保留完整候选列表
- 你现在处理后的训练集候选长度，来自 train_impressions.parquet，大概是：

- 平均 candidate_len ≈ 5.21
- 中位数 5
- 最大 39

而原始不采样的训练集候选长度，在 data/raw/MINDsmall_train/behaviors.tsv 上统计大概是：

- 平均 ≈ 37.23
- 中位数 24
- 75 分位 51
- 90 分位 91
- 95 分位 118
- 99 分位 166
- 最大 299

这个差距非常大。

也就是说，你一旦不做负采样，训练候选长度会从平均 5,变成平均 37，而且长尾很重。


改了学习率 ReduceLROnPlateau：
- mode="min"
- factor=0.5
- patience=1
- min_lr=1e-6
- 前期先用 5e-4 正常学
- 如果 dev_loss 连续一个 epoch 没变好
- 下一轮学习率就减半
- 最低不会降到 1e-6 以下


## 最终 early stopping 分数

![[Pasted image 20260404224030.png]]

### Early stopping

看 selection_score

规则：
- min_epochs = 3
- patience = 3
- min_delta = 0.001

也就是：
- 至少跑 3 个 epoch
- 如果连续 3 个 epoch，selection_score 没提升超过 0.001
- 就停

### Best checkpoint
也看 selection_score
也就是：
- 哪个 epoch 的 selection_score 最高
- 就保存哪个 checkpoint


## 学习率调度器怎么办

你现在已经接了 ReduceLROnPlateau。  
我的建议是：

- **scheduler 继续看 dev_loss**
- **early stopping / best checkpoint 看 selection_score**

理由很简单：

- scheduler 是为了优化训练稳定性
- early stopping 是为了选最终模型