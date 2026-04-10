```
greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab['S'])
```
`tgt_vocab['S']` 是什么

在 `data.py` 里，目标侧词表里有单独一个符号 `'S'`，表示 Decoder 输入序列的起始符（README 里也写了：`S: decoding input 的起始符`）。  
`tgt_vocab` 是「词 → 编号」的字典，所以 `tgt_vocab['S']` 就是 起始符对应的整数 id（在你这份代码里是 6，因为 `tgt_vocab = {'P':0,'i':1,..., 'S':6, ...}`）。


步入
```
enc_outputs = model.encoder(enc_input)
```

enc_input.shape
torch.Size([1, 5])

enc_input
tensor([[1, 2, 3, 5, 0]], device='mps:0')

enc_outputs
tensor([[[-1.3657, -0.0560, 0.5138, ..., 0.4101, -0.9151, 0.0723], [-0.4888, -0.5553, 1.7630, ..., 0.1190, -0.0068, 0.4421], [-0.6326, -0.2387, 0.8457, ..., 0.2795, -0.7710, 0.2577], [-0.0580, -0.4087, 0.3602, ..., -0.3225, -0.1412, 0.9039], [-1.1829, -0.5619, 0.7210, ..., -0.3138, -0.2693, 0.3981]]], device='mps:0')

enc_outputs.shape
torch.Size([1, 5, 512])

dec_input
tensor([], device='mps:0', size=(1, 0), dtype=torch.int64)

```
next_token = torch.tensor([[next_symbol]], dtype=enc_input.dtype, device=enc_input.device)
dec_input = torch.cat([dec_input.detach(), next_token], -1) #
```
tensor([], device='mps:0', size=(1, 0), dtype=torch.int64)
next_token:tensor([[6]], device='mps:0')
dec_input:tensor([[6]], device='mps:0')

```
dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
```
```
dec_outputs = self.tgt_emb(dec_inputs)
dec_outputs = self.pos_emb(dec_outputs)
```
[1,1,512]

```
dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

```
输入的是dec_input:tensor([[6]], device='mps:0')，dec_input:tensor([[6]], device='mps:0')，不是embeddiing后的
```
def get_attn_pad_mask(seq_q, seq_k):
	batch_size, batch_size = seq_q.size()
	batch_size, len_k = seq_k.size() 
	pad_attn_mask = seq_k.eq(0).unsqueeze(1)
	res = pad_attn_mask.expand(batch_size, len_q, len_k)
	return res 
```
![[Pasted image 20260403173328.png|442]]

```
dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)#因果mask
#输入还是dec_input:tensor([[6]], device='mps:0')
```

```
def get_attn_subsequence_mask(seq):
	attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
	subsequence_mask = torch.triu(
	torch.ones(attn_shape, device=seq.device, dtype=torch.bool), diagonal=1
	)
	return subsequence_mask
```
![[Pasted image 20260403173942.png|453]]

```
dec_self_attn_mask = torch.logical_or(dec_self_attn_pad_mask, dec_self_attn_subsequence_mask)
```
![[Pasted image 20260403174224.png|385]]
```
dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
```
![[Pasted image 20260403174448.png|420]]

```
def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size() 
	batch_size, len_k = seq_k.size()
	pad_attn_mask = seq_k.eq(0).unsqueeze(1)
	res = pad_attn_mask.expand(batch_size, len_q, len_k)
	return res 
```
![[Pasted image 20260403181842.png|544]]
## 第一次进解码器
```
for layer in self.layers:
	dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
```
![[Pasted image 20260403182534.png|303]]
![[Pasted image 20260403182600.png|305]]
步入之后
```
dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
```
![[Pasted image 20260403182853.png|352]]
这里进的是多头注意力
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) # [batch_size, len_q , d_model]
	return self.layer_norm(output + residual) # output: [batch_size, len_q, d_model]
```
![[Pasted image 20260403183728.png|251]]
![[Pasted image 20260403183750.png|254]]![[Pasted image 20260403183807.png|257]]
步出之后执行下一步
```
dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
```
![[Pasted image 20260403184152.png|581]]

这里进的还是多头注意力机制
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) 
	return self.layer_norm(output + residual)
```
![[Pasted image 20260403185157.png|591]]

![[Pasted image 20260403185505.png|596]]
![[Pasted image 20260403185636.png|595]]
步出之后
```
dec_outputs = self.pos_ffn(dec_outputs)
```
![[Pasted image 20260403185831.png|423]]

```
def forward(self, inputs):
	residual = inputs
	output = self.fc(inputs)
	return self.layer_norm(output + residual)
```
![[Pasted image 20260403190056.png|434]]
## 第二次进layer
```
dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
```
![[Pasted image 20260403190534.png|563]]
步入函数
```
def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
	dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
	dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
	dec_outputs = self.pos_ffn(dec_outputs)
	return dec_outputs
```
再步入dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
![[Pasted image 20260403190917.png|372]]
这里是多头注意力
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) 
	return self.layer_norm(output + residual) 

```
![[Pasted image 20260403191624.png|339]]
![[Pasted image 20260403191848.png|261]]
![[Pasted image 20260403191904.png|265]]
![[Pasted image 20260403191927.png|311]]
步出再步入
```
dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
```
![[Pasted image 20260403192335.png|615]]
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) # [batch_size, len_q , d_model]
	return self.layer_norm(output + residual) #
```
![[Pasted image 20260403192710.png|433]]
![[Pasted image 20260403192725.png|437]]
![[Pasted image 20260403192745.png|445]]
步出之后执行
```
dec_outputs = self.pos_ffn(dec_outputs)
```
![[Pasted image 20260403193050.png]]
## 每个layer都一样，把解码器的表示依次送过 n_layers 层（你这边是 6 层）DecoderLayer，一层一层堆叠，越往后语义越“深”。
```
dec_outputs = model.decoder(dec_input, dec_input, enc_outputs)
projected = model.projection(dec_outputs)
prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
next_symbol = prob[-1].item()

```
![[Pasted image 20260403195009.png|671]]
![[Pasted image 20260403195129.png|255]]
![[Pasted image 20260403195550.png]]
## 生成下一个字，重新进入while
```
while flag:
	next_token = torch.tensor([[next_symbol]], dtype=enc_input.dtype, device=enc_input.device)
	dec_input = torch.cat([dec_input.detach(), next_token], -1)
	
	dec_outputs = model.decoder(dec_input, enc_input, enc_outputs) 
	
	projected = model.projection(dec_outputs) 
	prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
	next_symbol = prob[-1].item()
	if next_symbol == tgt_vocab['.']:
	flag = False
	print(next_symbol)
```
![[Pasted image 20260403200638.png|238]]
![[Pasted image 20260403200657.png|241]]
![[Pasted image 20260403202004.png|263]]
接下来再次执行这句
```
dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
[1,2].   [1,5]   [1,5,512]
```
进入decoder
```
def forward(self, dec_inputs, enc_inputs, enc_outputs):
	dec_outputs = self.tgt_emb(dec_inputs)
	dec_outputs = self.pos_emb(dec_outputs)
	dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)#padding mask
	dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)#因果mask
	dec_self_attn_mask = torch.logical_or(dec_self_attn_pad_mask, dec_self_attn_subsequence_mask)
	dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
	for layer in self.layers:
	dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
	return dec_outputs # dec_outpu
```

![[Pasted image 20260403210332.png]]
进入
```
dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs）
```
![[Pasted image 20260403210651.png]]
```
def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size() 
	batch_size, len_k = seq_k.size()
	pad_attn_mask = seq_k.eq(0).unsqueeze(1)
	res = pad_attn_mask.expand(batch_size, len_q, len_k)
	return res 
```
![[Pasted image 20260403211131.png|546]]
```
dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
```
![[Pasted image 20260403211253.png]]
```
def get_attn_subsequence_mask(seq):
	attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
	subsequence_mask = torch.triu(
	    torch.ones(attn_shape, device=seq.device, dtype=torch.bool), diagonal=1
	)
	return subsequence_mask
```
![[Pasted image 20260403211813.png|402]]
```
dec_self_attn_mask = torch.logical_or(dec_self_attn_pad_mask, dec_self_attn_subsequence_mask)
```
![[Pasted image 20260403212046.png|398]]
接下来是交叉注意力
```
dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
```
![[Pasted image 20260403212224.png]]
```
def get_attn_pad_mask(seq_q, seq_k):
	batch_size, len_q = seq_q.size() # 获取作为q的序列（句子）长度 2，5，解码是是2，6
	batch_size, len_k = seq_k.size() # 获取作为k的序列长度 2，5，解码时也是2，5
	pad_attn_mask = seq_k.eq(0).unsqueeze(1)
	res = pad_attn_mask.expand(batch_size, len_q, len_k)
	return res
```
![[Pasted image 20260403213315.png]]
![[Pasted image 20260403213440.png]]
进入layers
```
for layer in self.layers:
	dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
```
![[Pasted image 20260403213921.png]]
开始生成下一个字
```
def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
	dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
	dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
	dec_outputs = self.pos_ffn(dec_outputs)
	return dec_outputs
```
再进入
```
dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
```
进入多头注意力
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) # [batch_size, len_q , d_model]
	return self.layer_norm(output + residual)
```
![[Pasted image 20260403214915.png|244]]
![[Pasted image 20260403215032.png|429]]![[Pasted image 20260403215050.png|244]]
上三角 mask 定义的是 「从左到右生成」的注意力结构；训练时它避免偷看标签，推理时它避免并行算多位时让左边的位错误地依赖右边的位。
再进入交叉注意力,生成的和原来的做注意力
```
dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
```
![[Pasted image 20260403215813.png|598]]
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) # [batch_size, len_q , d_model]
	return self.layer_norm(output + residual) # output: [batch_size, len_q, d_model]
```
![[Pasted image 20260403220608.png|262]]![[Pasted image 20260403220624.png|458]]
8个头
![[Pasted image 20260403220727.png|253]]
跳出注意力
```
dec_outputs = self.pos_ffn(dec_outputs)
```
![[Pasted image 20260403220947.png|291]]

# 生成第三个字，进入layer
```
for layer in self.layers:
	dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
```
![[Pasted image 20260403221257.png|538]]
进入layer
```
def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
	dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
	dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
	dec_outputs = self.pos_ffn(dec_outputs)
	return dec_outputs
```
先进入第一个注意力
```
dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
```
![[Pasted image 20260403221612.png|384]]
	步入注意力,这里是原来生成的聚合
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) # [batch_size, len_q , d_model]
	return self.layer_norm(output + residual) # output: [batch_size, len_q, d_model]
```
![[Pasted image 20260403222925.png|249]]
8个注意力头，所以有8个mask
![[Pasted image 20260403223020.png|291]]
下一个注意力
```
dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
```
![[Pasted image 20260403223338.png|595]]
进入注意力
```
def forward(self, input_Q, input_K, input_V, attn_mask):
	residual, batch_size = input_Q, input_Q.size(0)
	Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
	V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
	attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
	context = ScaledDotProductionAttention()(Q, K, V, attn_mask)
	context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
	output = self.concat(context) 
	return self.layer_norm(output + residual) 
```

![[Pasted image 20260403223809.png|493]]
8和注意力头，所以8个mask
![[Pasted image 20260403223936.png|363]]
步出 然后进入while
先来这里
```
dec_outputs = model.decoder(dec_input, enc_input, enc_outputs)
```
![[Pasted image 20260403230854.png|238]]
```
projected = model.projection(dec_outputs)
prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
next_symbol = prob[-1].item()
```
projection内部·
```
def forward(self, input: Tensor) -> Tensor:
	return F.linear(input, self.weight, self.bias)
```
![[Pasted image 20260403231529.png|375]]
![[Pasted image 20260403231538.png|370]]
进入while的开头，把下一个token输入拼入
```
while flag:
	next_token = torch.tensor([[next_symbol]], dtype=enc_input.dtype, device=enc_input.device)
	dec_input = torch.cat([dec_input.detach(), next_token], -1) 
```
![[Pasted image 20260403232711.png|383]]
