参数分析
我看完你现在这版代码和数据分布后，给你一个**直接建议**：

- max-title-len = 24：**不用改**
- max-entity-len = 5：**当前 baseline 不用改**
- max-history-len = 50：**可以保留作为 baseline，但最值得单独试的改动是把它提到 75**

## 先说结论为什么

### 1. max_title_len = 24

这个我觉得**很合适**。我看了原始新闻标题 token 长度分布：平均：11.17
- 中位数：11
- 90% 分位：15
- 95% 分位：17
- 99% 分位：21
- 最大：58

这说明：
- 24 已经覆盖了 **99%+** 的标题
- 真正被截掉的只是极少数超长标题
- 再往上加，对信息增益不会太大，但张量会变长一点

### 2. max_entity_len = 5

这个参数在你**当前 baseline 里其实基本没影响**。

原因是你现在 train.py 里模型初始化还是：

- use_entities=False

而且你当前主线还是 title only。  
所以：

- 即使 preprocess.py 里保留了 entity
- 训练时也没真正用上

所以对你现在这版来说：

**改不改 5，几乎不会影响结果。**

不过我还是顺手看了原始实体长度分布：

- 平均：2.38
- 中位数：2
- 90% 分位：5
- 95% 分位：6
- 99% 分位：8
- 最大：30

这说明如果你以后要做 entity 版本：

- 5 是个偏保守但能用的值
- 如果要更认真用实体，我会更推荐你以后试 8

### 3. max_history_len = 50

这个是三者里**最值得你认真考虑**的。

我看了原始历史长度分布：

- 平均：32.54
- 中位数：19
- 75% 分位：42
- 90% 分位：78
- 95% 分位：109
- 99% 分位：190
- 最大：558

而你处理后的数据里：

- 90% / 95% / 99% / max 全都卡在 50

这说明：

**有相当一部分样本的历史确实被你截断了。**

也就是说：

- 50 不是“没碰到上限”
- 而是“已经在截很多长历史样本”

## 那 max_history_len=50 要不要改

我的建议是：

### 当前 baseline

**先保留 50**

因为你现在的用户编码器在 modelbaseline.py 里只是：

- UserEncoder = masked_mean_pool(history_news_vecs, history_mask)

这是一个**平均池化用户编码器**。

这意味着：
- 它不会区分最近点击和很早点击
- 历史太长时，信息反而可能被平均冲淡
- 所以不是历史越长越一定好

也就是说，在你这版简单架构下：
- 50 是个挺稳的 baseline 值
- 盲目拉到 100、200 不一定更好

---

### 如果你想做一个最有价值的单变量实验

我会建议你试：

- **max_history_len = 75**

而不是一下跳到 100+。

原因：

- 75 能覆盖更多 90% 分位附近样本
- 比 50 多一些历史上下文
- 但不会像 100+ 那样把平均池化用户向量冲得太稀
- 计算开销增加也还是线性的，可接受



## 负采样

**负采样比例**：通常负样本的数量会是正样本数量的若干倍。例如，一般设置负样本的个数是正样本的 5 到 10 倍。也就是说，如果每个用户和物品的交互数据中有1个正样本，可以选择5到10个负样本进行训练。这个比例可以根据实验的效果进行调整。我们采用的是1:8，negmax=24，只有大约 **4.8%** 的训练样本会被上限截断。

## 原始数据分析：
原始训练集：平均候选数大约 **37**，中位数 **24**，90 分位 **91**，最大 **299**
训练集里大约 **27.44%** 的 impression 有多个正样本，>2 个正样本：17,506，占 11.1528%，>5 个正样本：2,311，占 1.4723%，>=10 个正样本：360，占 0.2294%。
验证集里大约 **28.82%** 的 impression 有多个正样本。

现在是 request-level DataLoader：candidate_ids [B, K]，labels [B, K]，batch 内需要 padding 到同一个最大 K。  所以我们采用一个分桶策略，可以有效减少无效padding。
## 分桶

训练时候先epoch，再bucket，再batch。在同一个桶内部，从这个桶对应的 impression 列表里，按该桶自己的 batch size，一批一批切出来。
candidate_len ：一个 impression 里，最终保留下来的候选新闻总数。
平均 candidate_len ≈ 11，中位数9 ，75% 分位 14，90% 分位 24，最大59。

所以我们的分桶方式是先按候选数把样本分成 short / medium / long。
短桶：1~10，中桶：11~24，长桶：25+。
然后不同的桶用不同的batchsize- BCE / ListNetTop：8 / 4 / 2，Pairwise：4 / 2 / 1。
batch_size 表示一个 batch 里有多少个 impression / request。
 batch 里的“每一行”仍然是一整个 impression。

## batchszie大小对数据的影响
可以理解成每一步更新参数时，看了多少条样本才做决定。参数更新 1 次= 发生在 **1 个 batch 之后**。**同一个 batch 里的样本通常是一起并行算的**。
训练集总共有：156,965 个 impression，而 batch_size = 8，那一个 epoch 大概有：
156965/8≈19621156965/8≈19621个 batch。一个 epoch 大约会更新 **19,621 次参数**

### 小 batch

比如 batch_size=2： 这一小步只看了 2 个 impression。如果这 2 个样本刚好比较特殊
。算出来的梯度就会“带偏”，所以下一步更新方向会抖一些。
### 大 batch：
一次看了 32 个 impression，各种样本平均一下。梯度更接近“整体数据的真实方向”。


生成数据：
conda activate mind
cd /Users/lixiang/Desktop/MIND
PYTHONPATH=src python src/preprocess.py \
  --train-dir data/raw/MINDsmall_train \
  --dev-dir data/raw/MINDsmall_dev \
  --output-dir data/processed \
  --max-history-len 50 \
  --max-title-len 24 \
  --max-entity-len 5 \
  --title-only \
  --train-negative-sample-ratio 8 \
  --train-negative-sample-max-size 24
  
PYTHONPATH=src python src/train.py --loss-type bce --wandb-run-name bce-ratio8-cap24-bucket

PYTHONPATH=src python src/train.py --loss-type pairwise --wandb-run-name pairwise-ratio8-cap24-bucket

PYTHONPATH=src python src/train.py --loss-type listnet_top --wandb-run-name listnet-ratio8-cap24-bucket


## 特征

新闻侧原始字段在 preprocess.py (line 12)：
- news_id
- category：新闻类别特征
- subcategory：新闻子类别特征
- title （ title_tokens）：标题分词后的 token 序列
- abstract（abstract_tokens）：摘要分词后的 token 序列
- url
- title_entities + abstract_entities（entities）：知识库的实体编号


行为侧原始字段在 preprocess.py (line 23)：
- impression_id：
- user_id
- time：时间
- history：用户历史点击序列
- impressions

## 特征里的attention
对hisrory新闻和candidate新闻，history 新闻和 candidate 新闻都会先各自编码成新闻向量，然后再通过 target attention 生成 candidate-aware 的用户向量，最后再打分。
- history_ids -> history_news_vecs：
   类别 + 子类别 + 标题表示 + 摘要表示 + 实体表示 -> 投影成统一新闻向量
   title / abstract 编码用到了注意力：先做self-attention建模token顺序和上下文,再做attention pooling，把整段文本压成一个向量
- candidate_ids -> candidate_news_vecs：
  category+subcategory+title+abstract+entity -> 投影成统一新闻向量
  title / abstract 编码用到了注意力：先做self-attention建模token顺序和上下文,再做attention pooling，把整段文本压成一个向量

- history_news_vecs + candidate_news_vecs -> user_vecs
  这一步history和candidate的建模用到了注意力：query = candidate_news_vecs，
  key = history_news_vecs  value = history_news_vec。每一个 candidate 都会去“看”用户历史里哪些新闻和它最相关。针对不同的user的历史，生成了对应这个user的历史的特定candidate的向量表示。
  history_news_vecs [B, H, D] +  candidate_news_vecs[B, K, D] -> user_vecs[B, K, D]

- user_vecs + candidate_news_vecs -> logits

## 特征交叉
 candidate-history 的结构化统计交叉：
 为每个 candidate 计算一个固定长度显式交叉向量 explicit_cross_features [B, K, 7]，包含：
same_category_rate  ： 有效历史中与 candidate category 相同的新闻占比
same_subcategory_rate：有效历史中与 candidate subcategory 相同的新闻占比
same_cat_subcat_rate：有效历史中同时满足 category + subcategory 都相同的新闻占比
entity_overlap_count ： candidate 实体里有多少个实体曾在有效历史中出现
entity_overlap_rate_candidate：entity_overlap_count / candidate_entity_count
entity_hit_history_rate：有至少一个实体与 candidate 重合的历史新闻占比
entity_overlap_any ： 只要存在实体重合则为 1.0，否则为 0.0
看看后面能不能通过门控分成2组，类别加实体。


candidate_user_cosine ：cos(candidate_news_vec, user_vec)
candidate_history_mean_cosine：cos(candidate_news_vec, masked_mean(history_news_vecs))
candidate_best_history_cosine：max_j cos(candidate_news_vec, history_news_vec_j)
history_attention_max：当前 candidate 对有效历史的 target-attention 权重最大值
history_attention_entropy：对有效历史 attention 权重做**归一化熵**；有效历史长度小于等于 1 时取 0
mul_feat = user_vecs * candidate_news_vecs：高维特征
1. recent_same_category_rate_5  
    如果 history 顺序保留时间信息，就只看最近 5 条历史里，和 candidate 同 category 的比例。  
    意义：补“近期兴趣”信息。
    
2. recent_same_subcategory_rate_5  
    同上，但看 subcategory。  
    意义：比 category 更细。
    
3. recent_entity_hit_history_rate_5  
    最近 5 条历史里，有实体命中的历史新闻占比。  
    意义：你现在有全历史 entity overlap，但缺“最近命中”。
    
4. candidate_top3_history_cosine_mean  
    candidate 和所有有效 history 的 cosine，取 top-3 的平均。  
    意义：你现在只有 best_history_cosine，太尖了；这个更稳。
    
5. candidate_recent_best_history_cosine_5  
    只在最近 5 条历史里取最大 cosine。  
    意义：把“语义最相似”和“近期性”结合起来。
    
6. history_attention_top2_mass  
    当前 candidate 的 target attention 里，前两大权重之和。  
    意义：比单独 max 更稳，也比熵更直观。

后面看能不能吧cos换成点乘


## 特征筛选
用你现在训练后自动产出的 **L2 importance**
再加一个 **特征相关系数热力图**
再用 **PCA** 看主成分结构



## 主思路
收集 dev 上所有有效样本的explicit_cross_features -> X [N,12]。
再加一个 **特征相关系数热力图**
L2 importance：模型当前更依赖谁
correlation：哪些特征高度重复
PCA：整体信息集中在哪几组维度里


- k_means
收集 dev 上所有有效样本的：explicit_cross_features -> X [N,12]
每一列做标准化标准化，特征尺度不完全一样。
对这 12 个“特征向量”做聚类
每个 cluster 留 1-2 个代表，如果两个特征在大多数样本上变化趋势很像，它们就会被分到一个 cluster。

- PCA
收集 dev candidate 样本的：explicit_cross_features -> X [N,12]
对每一列做标准化
做 PCA看： explained variance ratio，各主成分 loading。
如果前 2 到 3 个主成分就解释了很大比例，这 12 个特征里有很多冗余，信息比较集中。
如果要保留解释性，就说明你可能可以删掉一些高度重复的特征。
比如 PC1 上权重特别大的都是：same_category_rate，same_subcategory_rate，same_cat_subcat_rate那说明这 3 个特征本质上在表达同一类信息。
哪些特征是“一组”，哪些特征可能重复。


- 阈值确定
在现有 feature 模型的显式特征重要性输出基础上，增加一套 **噪声对照阈值** 分析。
阈值方法固定为：**加入 3 个对照噪声特征**，训练后用它们的第一层 L2 norm 重要性确定阈值。
生成方式固定为：基于 impression_id + candidate 位置 + control 索引 的**哈希式伪随机标量**，取值范围固定到 [-1, 1]。
重要性仍基于 FeatureClickScorer.mlp[0] 第一层权重列的 L2 norm。
阈值固定定义为：threshold_l2 = max(l2_norm(__noise_control_1..3))

- 过滤法 （filter method）
基于特征和目标变量的统计关系进行快速筛选
Pearson / Spearman：能直接用，最省事。适合先粗看每个特征和点击标签的单变量关系。
Mutual Information：也能用，而且比线性相关更适合你这种可能有非线性关系的特征，我很推荐作为第二个粗筛指标。
Variance Threshold：可以用，但它只能告诉你“这个特征是不是几乎不变”，只能当预过滤。
Chi-Square：你这里不太合适直接用，因为你现在有不少特征不是纯非负离散量，像 cosine 可能是负的，直接做卡方不自然。

**第一阶段：只训练 1 次**

- 先跑 feature + all12，拿到一个 best checkpoint。
- 在这个固定模型上做 **dev-only permutation/shuffle importance**：
    - 每次只打乱一个显式特征维度
    - 不重新训练
    - 只重新跑一遍 dev evaluate
- 看 MRR / nDCG@10 / SelectionScore 掉多少。

这一步很省时间，因为：

- 只训练一次
- 后面 12 次只是推理，不是训练


# 到了 SENet 阶段怎么接


PYTHONPATH=src /opt/anaconda3/envs/mind/bin/python src/train.py --model-type feature --loss-type pairwise
PYTHONPATH=src /opt/anaconda3/envs/mind/bin/python src/train.py --model-type feature --loss-type pairwise --explicit-cross-features same_category_rate,candidate_user_cosine,history_attention_entropy
PYTHONPATH=src /opt/anaconda3/envs/mind/bin/python src/train.py --model-type feature --loss-type pairwise --disable-explicit-cross-features
- 默认不传参数时启用全部 12 维
- 用 --explicit-cross-features name1,name2,... 手动指定子集
- 用 --disable-explicit-cross-features 完全关闭显式交叉
等你做完筛选后，我建议把这些显式交叉和原来的 scorer 输入分成 3 组：

1. user/candidate 语义组
2. mul/abs_diff/dot 交互组
3. explicit_cross 显式统计组

然后再决定是：

- 对整个拼接后的大向量做 SENet
- 还是先按组做 Gate，再拼接




**先做组级 Gate，再做一个轻量 SENet**，但这是下一阶段的事。
--


pairwise我们的目的是让正负样本的得分拉的更开，有点像对比学习，能不能就是让他在特征表示空间里的拉的更开，就是每个正负样本都有一个embeding表示。

## 特征交叉
- **新闻内部交叉**：将同一条新闻的不同字段组合，例如类别与子类别、类别与关键 token 等，挖掘新闻内部结构关系。
- **新闻与用户兴趣交叉**：将候选新闻的字段与用户历史兴趣分布做交叉，体现用户对特定领域或实体的偏好。
- **跨模态交叉**：在文本向量和实体向量之间进行交叉，以捕捉语义与知识之间的互动

| 交叉项                | 说明与实现方法                                                                                                                       | 备注                             |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **类别 × 子类别**       | 将类别嵌入 (e_{cat}) 与子类别嵌入 (e_{sub}) 进行**元素乘积**和**拼接**，得到 ([e_{cat} \odot e_{sub}, e_{cat} \oplus e_{sub}])。后续 SENet 会学习不同通道的重要性。 | 交叉维度相对较小，能体现领域与细分领域之间的联系。      |
| **类别 × 关键 token**  | 从标题中选取前 _k_ 个高频 token（或通过 TF‑IDF 选取），取其嵌入平均值 (e_{tok})。将 (e_{cat}) 与 (e_{tok}) 进行元素乘积和拼接。                                     | 体现新闻主题与主要关键词之间的关联。 _k_ 可取 3～5。 |
| **子类别 × 关键 token** | 类似于上项，用子类别嵌入与关键 token 嵌入交叉。                                                                                                   | 在某些细分领域，关键词的重要性更强。             |
| **文本 × 实体**        | 如果启用实体特征，将标题/摘要文本嵌入（平均池化或轻量 Transformer 输出）与实体嵌入 (e_{ent}) 进行**逐元素乘积**和**差分**；例如 (e_{txt} \odot e_{ent},                      | e_{txt}-e_{ent}                |








**单任务 CTR 排序模型 + 用户历史序列编码 + 同 impression 内 pairwise RankNet loss + 少量显式交叉特征。**


## 冷启动，空history


- history_news_vecs 会是 shape 类似 [B, 0, D]
- user_encoder 不做 attention，直接保留 cold_start_user
- history_attention_weights 是空或全 0
- 显式交叉特征大多变成 0


## **特征重要性和特征剪枝，为什么这么复杂**

因为它不是普通 tabular 特征选择。
这里有两个难点：  
第一，特征不是直接喂给 MLP 的标量，而是 embedding 后再 concat 的。
所以你要看某个字段的重要性，不能只看一个标量权重，而要看**它对应那一段权重子矩阵**。

第二，新特征不是直接进 Expert，而是先过了一层重映射。

所以对新特征，真正的“有效权重”不是 Expert 第一层的某块矩阵本身，而是：
  **重映射层权重 × Expert 第一层权重**

材料里写的 A(x + By + Cz)，就是这个意思。老特征直接看 A 的子矩阵；新特征要把 B/C 和 A 乘起来再算重要性。
  
再配合 concrete dropout 和一个噪声对照特征去定阈值，最后把特征从大约 1000 剪到 800，AUC/GAUC 基本不掉。这个做法很有工程味，也很适合写进复现报告，因为它说明你不只是“堆特征”，还做了可解释的特征治理。材料里报告的这个结果，是项目很值得讲的一部分。

数据处理：

**behaviors.tsv每条样本保留成这种结构**
{
    "impression_id": ...,
    "user_id": ...,
    "time": ...,
    "history_news_ids": [...],     # 截断到最近 H 条
    "candidate_news_ids": [...],   # 这个 request 的候选列表
    "labels": [...],               # 0/1
}

第一版先用：

- max_history_len = 50
    
- max_title_len = 24
    
- max_entity_len = 5

- max_history_len = 30~50
    
- title only
    
- embedding_dim = 128
    
- batch_size = 8~16 impressions
    
- 先用 float32
    
- 不要先上 full pairwise + 很长 candidate list



  

MIND 上你可以做这些“近似替代”：

  

### **8.1 历史类目匹配**

  

对每个 candidate 计算：

- hist_same_category_count
    
- hist_same_subcategory_count
    
- last_click_same_category
    
- last_click_same_subcategory
    

  

### **8.2 实体重合**

  

从 title_entities / abstract_entities 里取实体 ID，做：

- candidate_entity_overlap_with_history
    
- candidate_entity_overlap_with_last_n_clicks
    

  

### **8.3 训练集统计特征**

  

只能用 **train_local** 统计，不能看 val_local/dev：

- news_click_count
    
- news_impression_count
    
- news_smoothed_ctr
    
- category_click_count
    
- category_smoothed_ctr
    

建议做一个平滑 CTR：smoothed_ctr = (clicks + 1) / (impressions + 5)


你最终的实验表，建议至少做这 5 行：

1. Popularity / CTR stats baseline
    
2. Pointwise BCE + mean history
    
3. + 显式交叉特征
    
4. + Pairwise rank loss
    
5. + Attention/GRU history encoder


原项目材料里重点讲的是 request 内排序能力和 GAUC 提升；在 MIND 上，你就把这个思想转成 **MRR / nDCG 的提升** 来讲。
MIND 数据中的新闻文本具有丰富的内容信息，可提取词袋、词向量或预训练模型表示；实体嵌入提供了外部知识；用户点击历史可视为序列信息。这些都可以作为模型输入，支持多模态特征交叉。

  

## **适用于 MIND‑Small 的特征交叉改进方案**

  

下面给出一个详细的实施方案，分为数据预处理、模型结构改进、损失函数设计和训练流程四部分。方案在 Baseline 架构基础上增加了特征交叉和序列建模模块，主要变化用粗体标注。

  

### **1. 数据预处理与特征扩充**

1. **新闻编码**：在 Baseline 的类别、子类别和标题 token 之外，再加入：
    
    - 摘要 token（必要时可截断到固定长度）；
        
    - 实体 ID 序列和实体嵌入（来自 entity_embedding.vec）；
        
    - 新闻长度、发布时间（可离散化）等结构化特征。
        
    
2. **用户实时序列**：保留每个用户的点击历史顺序，生成历史新闻向量序列和对应 mask。序列长度可截断在 50～100 之间；对于长序列，后续 MHTA 会自动筛选相关行为。
    
3. **Side Info 特征**：为实现 MHTA 中的 sideinfo，可以对用户行为序列和候选新闻的类别、子类别、实体等离散字段进行嵌入，并构建 multi‑field embedding，使注意力计算不仅考虑新闻向量，还考虑类别和属性匹配。用户 ID 可映射至用户嵌入，用于捕捉长期偏好。
    
4. **特征标准化与编码**：连续特征（如点击间隔）做归一化；离散特征映射到 embedding 矩阵；embedding 大小建议 32~64 维。
    

### **2. 模型结构**

  

下图展示了改进后的整体架构（文字描述），核心模块包括 **新闻编码器**、**用户序列编码器**、**特征交叉层**和 **打分器**。


#### **2.1 新闻编码器**

在 Baseline NewsEncoder 的基础上加入多模态融合：

1. **文本编码**：
    
    - 使用词嵌入+用轻量级 Transformer模块；
        
    - 对标题和摘要分别编码，然后拼接。
        
    
2. **实体编码**：利用实体 ID 查表获得实体嵌入entity，并通过均值池化得到实体向量。
    
3. **类别/子类别嵌入**：从离散映射表中获取向量。
    
4. 拼接上述向量，送入线性层归一化，得到新闻表示 $e_i$。
    

  

#### **2.2 用户序列编码器**

使用 **Multi‑Head Target Attention** 代替简单平均池化。对每条 impression 中的候选新闻 $c_j$，执行以下步骤：

1. **准备键和值**：将用户历史新闻表示序列 ${h_1, h_2, \dots, h_L}$ 与其 sideinfo 组成键和值矩阵；sideinfo 可以是类别、子类别、时间戳、行为类型等嵌入。
    
2. **准备查询**：将候选新闻表示 $e_j$ 与其 sideinfo 拼接，经线性投影得到查询向量 $q_j$。
    
3. **多头注意力计算**：对每个头 $h$ 计算 $\mathrm{softmax}( (q_j W_Q^h)(K W_K^h)^T / \sqrt{d_h} ) (V W_V^h)$，其中 $K$、$V$ 是历史序列，$W_Q^h, W_K^h, W_V^h$ 为可学习矩阵，$d_h$ 为每个头的维度。多头设计使不同头捕捉不同兴趣方面 。    
4. **聚合**：将各头输出拼接并通过线性层生成候选新闻对应的 **动态用户表示** $u_j$。此向量表示用户针对当前候选的兴趣激活。
    

  

注意：为了效率，可预计算历史新闻投影（参照 TWIN 的“通用检索”阶段）并缓存；在 MIND‑small 规模下，使用普通矩阵乘法即可。

  

#### **2.3 特征交叉层**
![[d6aa24317bc4ae79a0cdc886f228e21e 1.jpg]]
候选新闻向量 $e_j$、动态用户向量 $u_j$ 以及其他辅助特征输入到以下结构：

1. **SENet 层**：对拼接的特征矩阵 $X \in \mathbb{R}^{m\times d}$ 先全局平均池化获得通道统计 $s = \frac{1}{m}\sum_{k=1}^m x_k$，再经两个全连接层产生权重 $\alpha = \mathrm{sigmoid}(W_2,\mathrm{ReLU}(W_1 s))$，最后对每个特征通道做逐元素缩放 $x_k’ = \alpha \odot x_k$。SENet 通过学习重要性权重动态调整特征贡献 。
    
2. **Neural Gate**：将新闻和用户向量分别投影为门控向量 $g = \sigma(W_g[e_j \oplus u_j] + b)$，用于控制交叉特征流；最终交叉表示 $z = g \odot [e_j \oplus u_j] + (1-g) \odot (e_j \otimes u_j)$，其中 $\otimes$ 为 Hadamard 乘法，$\oplus$ 为拼接。该模块有助于自动选择适合的交叉方式。
    
3. **多层矩阵交叉网络（MMCN）**：连续堆叠若干层矩阵交叉单元，每层实现如下变换：
    
    $$x_{l+1} = x_l \cdot W_l + b_l + x_l \cdot U_l \cdot x_l^T,$$
    
    其中第一项为线性部分，第二项为矩阵交叉。通过多层叠加，可以捕捉高阶特征交互。
    
4. **打分 MLP**：将上述经过 SENet、Neural Gate 和 MMCN 的向量拼接后输入 MLP，输出每个候选新闻的点击 logits。MLP 层数可为 2–3，隐藏维度 128–256。原 Baseline 中的乘积、差和点积特征亦可作为输入，增强模型对显式交叉的表达。
    

  

### **3. 损失函数设计**

  

采用 **点对点二分类损失**和 **ListMLE** 加权融合：

1. **Pointwise BCE**：与 Baseline 相同，对每个候选新闻预测点击概率并计算二元交叉熵。该损失使模型保持良好的预测校准。
    
2. **ListMLE Loss**：对于每个 impression 中的候选列表 ${(e_{j},y_{j})}$，根据 Plackett‑Luce 模型将概率定义为按照模型打分依次抽取的概率 。具体来说，设模型打分为 $s_j$，则候选列表的对数似然为
    
    $$\log P(\mathrm{rank}|s) = \sum_{i=1}^K \log \frac{\exp(s_{\pi(i)})}{\sum_{t=i}^K \exp(s_{\pi(t)})},$$
    
    其中 $\pi$ 是按真实点击标签从高到低排序的索引序列。负对数似然即为 ListMLE 损失。
    
3. **联合训练**：总损失为 $\mathcal{L} = \lambda , \mathcal{L}_{\mathrm{BCE}} + (1-\lambda) , \mathcal{L}_{\mathrm{ListMLE}}$，其中 $\lambda \in [0,1]$ 为权重超参数。可通过实验在验证集上调节 $\lambda$。
    

  

### **4. 训练与评估流程**

1. **数据准备**：运行现有的 preprocess.py 生成新闻字典 (news_dict.parquet)、impression 列表等；并根据新增特征（摘要 tokens、实体等）扩展 build_news_feature_tensors 函数。将动态历史序列和 sideinfo 转换为张量。
    
2. **模型初始化**：
    
    - 加载扩充后的新闻特征矩阵并构建 embedding 表。
        
    - 初始化 Multi‑Head Target Attention 模块的查询/键/值投影矩阵，head 数可设 4–8。
        
    - 初始化 SENet、Neural Gate 和 MMCN 等模块。
        
    
3. **优化器**：使用 AdamW，学习率 1e‑3，配合权重衰减；可按需使用梯度裁剪 (clip norm 1.0)。
    
4. **训练策略**：
    
    - 对每个 batch，先通过新闻编码器获取候选和历史新闻表示。
        
    - 使用 MHTA 为每个候选构建用户动态表示。
        
    - 输入特征交叉层获得 logits，计算 BCE 和 ListMLE 损失并反向传播。
        
    - 每个 epoch 结束后在验证集评估 AUC、GAUC、MRR、nDCG 等指标；选取最优模型。
        
    
5. **推理和排序**：上线时只需保存新闻和实体嵌入表并缓存用户历史序列的投影结果。实时请求时，先用 MHTA 激活用户兴趣，再根据 logits 排序并返回 Top‑k 新闻。
    

  

## **可能的改进方向**

- **SimTier/聚类压缩**：对于长历史序列，可借鉴 TWIN 中的聚类策略，将长序列聚合成若干 cluster 表示。尤其是在 MIND‑large 数据集上，可以首先对序列进行 K‑Means 聚类，然后采用 cluster embedding 进行 MHTA。
    
- **预训练语言模型**：用轻量级预训练模型（如 DistilBERT）对新闻文本进行编码，有助于捕捉上下文语义；但需注意计算开销。
    
- **知识图谱增强**：利用新闻实体与知识图谱中的关系，构建 Graph Neural Network 或关系注意力，以更好地理解新闻之间的关联。
    

  

## **总结**

  

针对用户提供的 MIND‑small 数据集，可以将美团排序模型中采用的特征交叉和序列建模策略迁移应用：

1. 通过 **SENet** 动态调整不同字段的重要性，并结合 **Neural Gate** 和 **多层矩阵交叉网络** 捕捉新闻和用户特征的高阶交互 。
    
2. 采用 **Multi‑Head Target Attention** 根据候选新闻对用户历史序列进行动态激活，使模型关注与候选相关的行为 。
    
3. 采用 **Pointwise + ListMLE** 联合损失，用 BCE 保证概率预测准确，用 ListMLE 优化列表整体排序 。
    

  

该方案充分利用了 MIND‑small 中的新闻内容和用户行为信息，能够显著提升点击率/排名模型的离线 AUC 和在线指标。实现时可在现有 Baseline 代码结构中添加新的编码器和损失函数模块，按照上述流程进行训练和评估。




