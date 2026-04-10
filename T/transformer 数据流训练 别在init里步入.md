enc_inputs[2,5]
![[Pasted image 20260402151253.png|448]]

0是padding，所以
![[Pasted image 20260402151854.png]]

res = pad_attn_mask.expand(batch_size, len_q, len_k)
![[Pasted image 20260402152358.png|475]]
- `res[b, i, j] == True`：在 batch `b` 里，第 `i` 个 query 对第 `j` 个 key 的注意力要关掉（后面在 `ScaledDotProductionAttention` 里会把对应 `scores` 设成 `-1e9`）。
enc_outputs = layer(enc_outputs, enc_self_attn_mask)
![[Pasted image 20260402153526.png]]
![[Pasted image 20260402153542.png]]
enc_ouputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
![[Pasted image 20260402153707.png|231]]
就是input_Q, input_K, input_V
![[Pasted image 20260402154157.png]]
![[Pasted image 20260402154526.png]]
![[Pasted image 20260402154548.png|308]]8个
![[Pasted image 20260402155551.png|407]]
d_K：先把 512 维 的 (Q,K,V) 线性投影出来，再按 `n_heads = 8` 切成 8 份，每一份的维度64
这样 8 个头 × 每头 64 维 拼回去仍是 512，和 `d_model` 对齐。
![[Pasted image 20260402155608.png]]
![[Pasted image 20260402155632.png]]
「第 i 个 query 位置对第 j 个 key 位置的打分」
![[Pasted image 20260402155932.png]]
![[Pasted image 20260402160140.png]]
![[Pasted image 20260402160248.png]]
![[Pasted image 20260402161245.png]]
![[Pasted image 20260402161337.png]]
enc_ouputs = self.pos_ffn(enc_ouputs)#前馈
![[Pasted image 20260402162626.png]]
encoder不需要用到标签，Encoder 的职责是「只根据源语言句子抽表示」，不需要知道英语参考答案；  
标签只通过 Decoder 的输入/输出和最后的 loss 参与训练，梯度会穿回去更新 Encoder 和 Decoder 的参数，所以 Encoder 间接受到标签监督，但 前向里不会把标签张量喂给 Encoder
![[Pasted image 20260402163236.png]]
decoder里面才有标签交叉注意力里 Q 来自 Decoder，K/V 来自 Encoder 输出；mask 要挡的是 K 侧无效位置，无效性由 源 token id 是否为 padding 决定，所以 key 序列的「pad 信息」来自 `enc_inputs`，`dec_inputs` 只用来提供 query 长度并做形状上的 `len_q`。
![[Pasted image 20260402181902.png]]
![[Pasted image 20260402182112.png]]
![[Pasted image 20260402182118.png]]
![[Pasted image 20260402182315.png]]
![[Pasted image 20260402182526.png]]
![[Pasted image 20260402182621.png]]
![[Pasted image 20260402182712.png]]
![[Pasted image 20260402182727.png]]
![[Pasted image 20260402182920.png]]
![[Pasted image 20260402183040.png]]
![[Pasted image 20260402183209.png]]
![[Pasted image 20260402183350.png]]
![[Pasted image 20260402183634.png]]
![[Pasted image 20260402183644.png|387]]
交叉注意力里 Q 来自 Decoder，K/V 来自 Encoder 输出；mask 要挡的是 K 侧无效位置，无效性由 源 token id 是否为 padding 决定，所以 key 序列的「pad 信息」来自 `enc_inputs`，`dec_inputs` 只用来提供 query 长度并做形状上的 `len_q`。
```
return res # return: [batch_size, len_q, len_k]
```
![[Pasted image 20260402184033.png|592]]
![[Pasted image 20260402184202.png]]
注意上面都是dec
![[Pasted image 20260402184724.png]]
attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
![[Pasted image 20260402184909.png|467]]
8个 维度是[2,8,6,6]
context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
解码：context: [batch_size, n_heads, len_q, d_v][2, 8, 6, 64]
scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
scores.masked_fill_(attn_mask, -1e9)
![[Pasted image 20260402190027.png|271]]
 6是目标长度，`5` 是源长度；
 attn = nn.Softmax(dim=-1)(scores)
 ![[Pasted image 20260402191132.png]]
 ![[Pasted image 20260402191203.png]]
 第 `b` 个样本、第 `h` 个头 里，第 `i` 个 query 位置 对 第 `j` 个 key 位置 的 未归一化相似度
 **Decoder 掩码自注意力（`dec_self_attn`）**
调用大致是：`MultiHeadAttention(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)`  
这里的 `dec_inputs` 是 tensor：已经是 `tgt_emb + pos_emb` 之后的 `[B, tgt_len, d_model]`（变量名在 `DecoderLayer` 里可能叫 `dec_inputs`，实为当前层输入表示）。![[Pasted image 20260402191750.png]]
所以 `len_q = len_k = tgt_len`（你例子里是 6×6），维度一样。和上三角因果 mask 匹配的是 这一种。

**Decoder–Encoder 交叉注意力（`dec_enc_attn`）

调用大致是：`MultiHeadAttention(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)`  
（第一个参数是当前 decoder 在「做完自注意力之后」的张量。）
![[Pasted image 20260402191912.png]]
所以 `scores` 是 `[B, heads, tgt_len, src_len]`，例如 6×5：行是「译文的第几步」，列是「原文第几个词」。交叉注意力正是在做 「每个目标位置对齐到整段源序列」。
维度不一样的原因： physically 就是 两句话长度不同——Query 链路上是 目标长度，Key/Value 链路上是 源长度；交叉注意力正是在做 「每个目标位置对齐到整段源序列」。


**和线性层的关系（简要）**
`MultiHeadAttention` 里：
- `Q = W_Q(input_Q)`，`input_Q` 来自 Decoder → 时间维 `len_q`。
- `K = W_K(input_K)`，`input_K` 来自 Encoder 输出（交叉时）→ 时间维 `len_k`。
`Q @ K^T` 把时间维和 `d_k` 收缩掉，得到 `[len_q, len_k]`，所以 不等长 很正常。


context = torch.matmul(attn, V)
![[Pasted image 20260402192922.png|289]]
![[Pasted image 20260402192830.png|284]]
交叉注意力（`dec_enc_attn`）
 Q：目标侧第 `i` 步；K、V：源句每个位置 `enc_outputs`。
context[i]`：当前译文的第 i 步 从 整句原文 里，按 `attn` 分配的权重，加权读出来的一个向量。这一步翻译最该对齐原文哪些词、各拿多少信息」 都体现在 `attn` 里，`context` 就是把这一步从原文里读出来的摘要向量。

context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
output = self.concat(context)
![[Pasted image 20260402194008.png|252]]
```
def forward(self, input: Tensor) -> Tensor:

return F.linear(input, self.weight, self.bias)
```
![[Pasted image 20260402194124.png|225]]
dec_logits = self.projection(dec_outputs)
对最后一个维度 做 `512 → tgt_vocab_size`（你工程里是 9） 的仿射变换。
![[Pasted image 20260402194641.png|268]]
dec_logits.view(-1, dec_logits.size(-1))
![[Pasted image 20260402195014.png|404]]
