### **1. 混合损失与滞后模型的提示**

在 train_physres_ablation.py 中，我们引入了一个新的混合损失函数 compute_mixed_temporal_loss，它将指数衰减 MSE、均匀加权 MSE 和尾部 10 步的 MSE 按 0.5/0.3/0.2 加权组合 。这样可以让模型不仅关注短期误差（如 pos_h1、pos_h10），还重视长时步 h50 的表现。该损失通过 --loss-type mixed 激活。


  

### **2. 动态 GRU + 滞后初值 + 残余力的提示**

  

**滞后初值预测** **u_init_head**：原始滞后模型假设窗口开始时电机有效输入 u_eff 与 u_raw 相同，这在实际中可能偏离真实状态，导致长期漂移。在模型初始化时加入一个小的 MLP u_init_head，输入初始状态 x0 和 u0，预测一个初始偏置 δu0，以此修正 u_eff_prev_real = u0_real + δu0。
输入12+4=16
$$ \begin{aligned} \boldsymbol{x}_{\text{norm}} &\in \mathbb{R}^{(B, 12)} = \left[ p,\ v,\ \text{so3\_log},\ \omega \right] \\ u_{0,\text{norm}} &\in \mathbb{R}^{(B, 4)} = u_{\text{seq}}[:, 0, :] \end{aligned} $$
，输出u0_norm 4维。




1. **残余力输出** **force_head**：目前残差 MLP 输出的是状态增量 Δx，而论文指出未建模的主要是横向推力和低频扭矩。改为预测机体坐标系下的残余力 Δf_b，并在物理积分后施加这股力得到更新状态。这样网络直接学习缺失的动力学而不是积累状态差分，有助于减少 long-horizon 漂移。 
2. 输入维 hidden_dim：在 forward 里接的是 GRU 隐状态 h（与父类里的 gru_hidden_dim 一致，默认 64）。输出3 维：机体坐标系下的残差外力Δf_b，交给物理里的 apply_force
    $$ \boldsymbol{f}_{\text{meas, body}} = m \cdot \text{aux\_seq}[..., :3] $$ $$ \text{thrust} = K_t \cdot \left( \Omega_1^2 + \Omega_2^2 + \Omega_3^2 + \Omega_4^2 \right) $$ $$ \boldsymbol{f}_{\text{phys, body}} = \left[ 0,\ 0,\ \text{thrust} \right] $$ $$ \boldsymbol{f}_{\text{target, seq}} = \boldsymbol{f}_{\text{meas, body}} - \boldsymbol{f}_{\text{phys, body}} $$ $$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{base}} + \beta_{\text{force}} \cdot \text{SmoothL1Loss}\left( \boldsymbol{f}_{\text{pred, seq}}, \boldsymbol{f}_{\text{target, seq}} \right) $$
![[Pasted image 20260401223032.png|496]]
  
  ## GRU
## GRU

残差网络的输入特征为 $$ \boldsymbol{x}_{\text{in, Res+GRU}} = \left[ x_{\text{norm}},\ u_{\text{raw}},\ u_{\text{eff}},\ u_{\text{raw}} - u_{\text{eff}},\ h \right] $$；与无 GRU 版本不同，无 GRU 版本不含隐状态 h，输入维度为 state_dim+12。 
h形状 `(B, hidden_dim)`，作为 第一步的 GRU 隐状态。


此外，我们让 alpha_head 依赖于$[x_\text{norm}, u_\text{raw\_norm}, u_\text{eff\_norm}, (u_\text{raw\_norm} - u_\text{eff\_norm}), h]$ 动态生成滞后系数 α_t，并用 GRU 隐状态 h 记忆历史。

为了训练稳定，还提出将参数分为两组：u_init_head、alpha_head 用较大学习率（如 3e‑4），其余用较小学习率（1e‑4），并可为残余力预测加一个辅助损失项。最后建议在 build_model 中新增 "lag_gru_force" 变体，实例化上述新模型，并在 compute_loss 中同时计算混合 MSE 和残余力的 Huber/MSE 损失。



## PhysRes-GRU-Force
2. - 在上述基础上再加入 GRU 隐状态记忆和动态滞后系数 α_t，每个时间步根据当前状态、原始控制、滞后控制以及隐状态生成新的 α_t 用于更新电机有效输入，GRU 隐状态再与这些特征拼接后输入残差网络。这使模型能够建模更长的时序依赖。



- **可学习的滞后而不反传物理**。保留 lag 层 u_eff = alpha*u_eff_prev + (1-alpha)*u_raw，；
- 引入一个小的门控网络 g(x_t,u_t) 输出 alpha_t，并把 alpha_t 输入到 residual 网络，让残差学会在不同 alpha 下修正误差。这样 alpha 依旧有梯度.
    
- **为 lag 层单独设较大学习率**。滞后系数只有 1 或 4 个参数，用与残差相同的 1e‑5 学习率过小。建议在优化器中为 lag_layer.logit_alpha 设置 lr=3e-4，其余参数仍用 1e-4 或 1e-5。
    
- **改进滞后初值**。不要直接用当前 u_raw 做 u_eff_prev_real，而是让模型预测一个初值偏差，如 u_eff_prev_real = u_raw + δu_init(x0,u0)，δu_init 可以是一个小的两层 MLP。这可以减少整个窗口的累积漂移。
    
- **使用时序 GRU 学习残余力而非直接学 Δx**。论文指出未建模的主要是低频力和横向扭矩，而你的残差网络学的是位移增量，难以约束长时漂移。建议改成如下结构：  
- 论文已经明确指出 missing physics 里有 body-frame x/y lateral force。学 Δf_b 比学 Δx 更直接。这种 “PhysRes‑GRU‑Force” 结构既利用了发现的滞后效果，又通过 GRU 记忆长期误差。可以先只输出残余力 Δf_b 来提升 pos/vel 指标，再加残余转矩预测。

不是学 Δx，而是学**带记忆的 body-force residual**；先只攻你最需要的 pos / vel。

核心思路：
```
h = h_init(x0)
u_eff = u0 + du0_head(x0, u0)

for t in range(H):
    alpha_t = sigmoid(alpha_head(torch.cat([x_t, u_t, h], -1)))
    u_eff = alpha_t * u_eff + (1 - alpha_t) * u_t

    # 物理分支
    x_phys = phys_step(x_t, u_eff)

    # 时序残差分支
    h = GRUCell(torch.cat([x_t, u_t, x_phys], -1), h)
    df_b = force_head(h)          # R^3, body-frame residual force
    x_{t+1} = phys_step_with_force_residual(x_t, u_eff, df_b)
```
- physics 还是主干，GRU 只学 correction，不容易像纯黑箱那样 long-horizon drift。
- 我建议的第一版配置很具体：
- hidden size：64，不够再上 128
- GRU layers：1  
- residual output：先只输出 Δf_b (3)  
- loss：上面那套 long-horizon-aware loss
- aux：Δf_target 的 Huber loss，权重 0.1 ~ 0.2
- optimizer：
    - GRU + heads：1e-4
    - alpha_head / du0_head：3e-4    
    - weight decay：1e-5
- gradient clip：1.0
    
- **放大 batch_size**。当前默认 batch_size=256 。如果显存足够可以尝试 512 或 1024，这样每步能更充分利用 GPU 并减少 epoch 数。因为 LSTM/GRU 模型在时间维度上按 50 步 unroll，本质属于序列并行，增加 batch size 对收敛影响不大
  

先把 sim_pos、sim_vel、pos_h50、vel_h50 拉上去。

等这版稳了，再加一个 torque_head(h) -> Δτ_b 去攻 rot/omega。


 ：当前训练脚本在验证阶段只根据总损失 avg_valid_metrics["loss_total"] 来保存最佳模型 。如果你想更关注长时性能，建议增加一个开环 50 步的评测函数，在每个 epoch 验证后计算 pos_h50、vel_h50、sim_pos、sim_vel 等指标，并用加权公式 \frac{\text{pos\_h50}}{0.115} + \frac{\text{vel\_h50}}{0.521} + \frac{\text{sim\_pos}}{2.44} + \frac{\text{sim\_vel}}{10.14} 来决定是否更新最佳模型。