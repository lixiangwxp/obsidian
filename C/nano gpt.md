我根据官方 README 和苹果开发者文档整理了一份详细的操作说明，按照终端命令顺序解释了在 Mac M4 Pro 上搭建 nanoGPT 环境的全过程，包括 Python 环境的创建、安装支持 Metal GPU 的 PyTorch、克隆源码和训练小模型等步骤，并说明了各参数的作用、MPS 后端的局限与验证方法。

您可以在以下文档中查看全部内容：

# Mac M4 Pro 上安装与运行 nanoGPT 的详细步骤

以下说明结合了 nanoGPT 官方 README 文件和 Apple Developer 文档，给出在 Mac M4 Pro（Apple Silicon）上配置训练环境并运行 nanoGPT 的具体操作指令。步骤按照实际终端操作顺序组织，并对每个命令进行了说明，便于理解原因和作用。

## 1. 系统与工具准备

1. **操作系统要求**：nanoGPT 依赖 PyTorch 的 Metal 性能着色器（MPS）后端，官方要求 macOS 12.3 及以上版本且设备为 Apple Silicon（M 系列）。请确保系统符合条件。
2. **安装 Xcode 命令行工具**：MPS 后端需要编译工具链，运行：
    
    xcode-select --install
    
    成功安装后可以提供必要的编译工具。
    
3. **可选：安装 Homebrew**：Homebrew 是 macOS 的包管理器，方便安装 Python 等工具，如果尚未安装，可以执行：
    
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    

## 2. Python 环境配置

### 

### 2.2 使用 Conda（可选）

如果习惯使用 Conda，可以先安装 Miniconda，然后创建新环境并安装 PyTorch：

1. **下载并安装 Miniconda（ARM 版本）**：
    
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  
    sh Miniconda3-latest-MacOSX-arm64.sh
    
2. **创建并激活环境**：
    
    conda create -n nanogpt python=3.10  
    conda activate nanogpt
    
3. **安装 PyTorch (nightly)**：
    
    conda install pytorch torchvision torchaudio -c pytorch-nightly
    
4. **安装其它依赖**：
    
    pip install numpy transformers datasets tiktoken wandb tqdm
    有冲突降级就可以
    

使用 pip 或 conda 安装完成后，都可以使用上述验证脚本检查 MPS 是否可用。

## 3. 克隆 nanoGPT 源码并准备数据

1. **克隆项目**：
    
    git clone https://github.com/karpathy/nanoGPT.git  
    cd nanoGPT
    
2. **下载并准备示例数据集**：nanoGPT 的 README 建议使用字符级 Shakespeare 数据集作为入门练习。运行数据准备脚本会下载数据并生成二进制格式的训练集和验证集：
    
    python data/shakespeare_char/prepare.py
    
    该命令会在 `data/shakespeare_char` 目录下生成 `train.bin` 和 `val.bin`。
    

## 4. 训练小模型（适合 Mac M4 Pro 调试）

nanoGPT 的默认配置针对大型 GPU。在只有 Macbook 的情况下，应将模型尺寸和训练参数调低。以下指令来自 README 的推荐，并针对 Apple Silicon 使用 `--device=mps` 以利用内置 GPU。

python train.py config/train_shakespeare_char.py \  
  --device=mps \  
  --compile=False \  
  --eval_iters=20 \  
  --log_interval=1 \  
  --block_size=64 \  
  --batch_size=12 \  
  --n_layer=4 \  
  --n_head=4 \  
  --n_embd=128 \  
  --max_iters=2000 \  
  --lr_decay_iters=2000 \  
  --dropout=0.0

**参数解释：**

- `--device=mps`：指定使用 Apple Silicon 的 Metal GPU 后端。README 提到在支持 MPS 的 PyTorch 版本中添加此参数可以利用芯片上的 GPU，训练速度提升 2‑3 倍。
- `--compile=False`：关闭 PyTorch 2.0 的编译功能，避免在 MPS 上出现兼容性问题。
- `--eval_iters`：每多少迭代评估一次损失，减小值可以加快训练进度，但评估更噪声。
- `--block_size`、`--batch_size`、`--n_layer`、`--n_head`、`--n_embd` 等参数控制模型大小和每次迭代的数据量，这些设置将模型缩小到 4 层、4 头、128 维嵌入，适合内存较小的机器。
- `--max_iters` 与 `--lr_decay_iters`：设置训练的迭代次数及学习率衰减长度。
- `--dropout`：由于网络较小，regularization 可以降低或设置为 0。

训练过程中会将检查点保存在 `out-shakespeare-char` 等目录中。由于 Macbook 的训练速度有限，全文训练可能需要数分钟到十几分钟；可以随时按 `Ctrl+C` 停止提前查看结果。

## 5. 生成文本

训练结束或中途停止后，可以使用 `sample.py` 从保存的模型中生成文本示例。由于部分操作（如 `torch.topk`）在 MPS 上尚未完全支持，采样时建议使用 CPU：

python sample.py --out_dir=out-shakespeare-char --device=cpu

如需自定义起始文本，可修改 `sample.py` 中的 `start` 变量。



## debug


```
def get_batch(split):
    #把 train.bin / val.bin 里的一长串 token id，随机切成很多段长度为 block_size 的小序列，做成输入 x 和目标 y。
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #随机采样 batch_size 个起点。
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

![[Pasted image 20260407222114.png]]




可以，这张图其实已经把 get_batch 的核心都暴露出来了。我按你现在这个粒度给你过一遍。

**data**  
你看到的是：
- 一个 memmap(...)
- shape=(1003854,)
- dtype=uint16

它代表：整个训练集的 token 流是一维的，一共有 1003854 个 token，每个数字是一个 token id。因为你现在是 shakespeare_char，这里的 token 基本可以理解成“字符编号”。
所以 data 不是一句一句文本，而是这种感觉：`[18, 47, 56, 43, 56, 43, ...]`
整本训练文本被编码成一长条数字。
data.shape = (1003854,)它表示：这是一个一维数组，长度是 1003854。
这里逗号很关键，(1003854,) 表示“1 维 shape”。  
**ix**  ：tensor([841669, 601862, 33636, 904609])`

它代表：这次 batch 从整条 token 流里随机选出的 4 个起点，现在 batch_size=4
也就是： 第 1 条样本从 data[841669] 开始切，第 2 条样本从 data[601862] 开始切，
        第 3 条样本从 data[33636] 开始切，第 4 条样本从 data[904609] 开始切
        所以 ix 本质上是“抽样到的起始下标”。

**x**  你看到的是一个 4 x 64 的整数张量。
它代表：一个 batch 的输入 token，一共 4 条样本，每条样本长度 64
        x.shape == [4, 64]，第一维 4 是 batch size，
        第二维 64 是 block size / context length
从语义上说：
- x[0] 是第 1 条训练样本
- x[0, 0] 是第 1 条样本的第 1 个 token
- x[0, 1] 是第 1 条样本的第 2 个 token

你看到里面很多像 43, 1, 47, 57... 这样的数字，不用现在去背它们对应什么字符。  
这时候你只需要知道：
- 它们是 token id
- 之后会送进 embedding 层

**block_size = 64**  
它代表： 模型一次最多看 64 个 token 的上下文，也是这里每条训练样本的长度
        所以这里 x 每一行长度都是 64，不是巧合，是配置决定的。

**x.shape = torch.Size([4, 64])**  
它表示：x 是一个二维张量，4 条样本。每条 64 个 token

这就是语言模型输入最常见的形状：`[batch_size, sequence_length]`

**y.shape = torch.Size([4, 64])**  
它代表标签张量 y 的形状和 x 一样。
语义上：x 是输入，y 是目标答案

但 y 不是随便来的，它是 x 向右错一位得到的。

- 给你 43，预测 1
- 给你 43,1，预测 47
- 给你 43,1,47，预测 57

```
logits, loss = model(X, Y)
```
步入
```
def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```
步入block
```class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
步入attn
```def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```

![[Pasted image 20260407231120.png|206]]![[Pasted image 20260407231155.png|278]]


步出，加步出到
```
for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        #(x.shape=[4, 64, 384])

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
```
![[Pasted image 20260407232224.png|452]]



