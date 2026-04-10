LOSS改进
![[Pasted image 20260401164518.png|555]]


### **为什么要在损失中使用 SO(3) 测度**
由于 dataset.py 和 PhysResQuadModel 的流程都在**标准化空间**里训练和预测，而 geodesic loss 的旋转向量必须有真实物理尺度，所以**最稳妥的做法是：先反标准化，再对 [6:9] 的 rotvec 计算 geodesic loss**。
![[Pasted image 20260329162427.png|345]]
论文的评估指标并不是欧式差，而是用四元数的相对旋转角度${ e_R}$。为了让训练目标与评估一致，我们在物理+残差模型上采用混合损失：先用标准的 MSE 学习位置、速度、角速度，再对反标准化后的旋转向量计算 geodesic 损失。旋转向量${ \mathbf{r}}$ 由四元数通过对数映射获得。这种损失比单纯的 MSE 更能约束姿态预测，从而提升旋转精度。




## . 仓库里有哪些「损失算子」

| 模块                      | 做什么                                                                                                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WeightedMSELoss`       | 序列 `(B,N,D)` 上，按时间 `exp(-λ·t)` 加权后的 全维 MSE（在传入张量的数值空间里算，消融里一般是 归一化后的状态）。                                                                                                                 |
| `WeightedGeodesicLoss`  | 12 维状态拆开：p / v / ω 的加权 MSE + `r` 的 SO(3) 测地角平方（先在 反标准化后的「物理空间」 上算几何项时由 `CompositeAblationLoss` 负责 denorm）；再按 `w_pos, w_vel, w_omega, w_rot` 线性组合。                                        |
| `CompositeAblationLoss` | `WeightedMSELoss`（归一化空间） + 可选 `WeightedGeodesicLoss`（反标准化后整条 12 维，但 geo 里常把 `w_pos,w_vel` 置 0 只强调旋转相关） + 可选 辅助头 `SmoothL1`（默认）。子项用 `β_geo`、`β_aux` 做 加权平均式合成 `total`（见 `losses_ext.py`）。 |
| `DenormRotGeodesicLoss` | 只对 反标准化后的 `r` 做与 `WeightedGeodesicLoss` 里 同套路的测地损失；当前 `train_physres_ablation.py` 的 `build_criterion` 里没有用它，属于可单独接的模块。                                                                   |
| 其它                      | `QuadStateMSELoss`、`ScaledMSELoss` 等在其他脚本或旧流程里；主 ablation 线不依赖它们。                                                                                                                        |
CompositeAblationLoss 如果设置 use_geo=True，会先用 denorm() 把 pred_seq_norm、true_seq_norm 反归一化，然后调用一个内部的 WeightedGeodesicLoss 来计算姿态和角速度的几何误差 。这相当于手动完成了 DenormRotGeodesicLoss 所做的事情，只是它还保留了其他维度的加权 MSE。也就是说，**你们实际使用的是反归一化后的几何损失**，只是通过组合损失框架实现的。

## 2. `train_physres_ablation.py` 里实际有几种「组合方式」

由 `--variant` 和 `--loss-type` 共同决定。

### A. 只用「纯时间域 MSE」（`CompositeAblationLoss` 不用）

对这些 variant：`baseline`，`lag`，`lag_gru`（以及 `lag_gru_force` 的主序列部分，见下）。

- `criterion = WeightedMSELoss(lambda_=0.1)`，只做 归一化序列上的指数加权 MSE。

`--loss-type`（仅对上面这类 `uses_plain_temporal_loss` 生效）：

- `exp`（默认）：就是上面的 单一 `WeightedMSELoss(0.1)`。
- `mixed`：`criterion` 置为 `None`，改用 `compute_mixed_temporal_loss`：  
    `0.5×WeightedMSELoss(λ=0.03) + 0.3×全局均匀 MSE + 0.2×最后 10 步 MSE`。

### B. `CompositeAblationLoss`（MSE ± 几何 ± 辅助）

对 `geo`，`lag_geo`，`full`：

- 基底永远是 归一化空间里的 `WeightedMSELoss(lambda_mse=0.1)`。
- `geo` / `lag_geo` / `full`：`use_geo=True` → 再加 反标准化后的 `WeightedGeodesicLoss`，权重系数 `--beta_geo`（默认 0.01）。
- 仅 `full`：`use_aux=True` → 再加辅助头 `SmoothL1`，系数 `--beta_aux`（默认 0.05）。

几何子损失内部还可调 `--w_rot`、`--w_omega`（传给 `WeightedGeodesicLoss`）。

### C. `lag_gru_force` 专用
`lag_gru_force` 是 固定用 mixed + force，脚本里还会把 `loss-type` 强制成 `mixed`（约 605–607 行）。

- 主序列：`mixed` 时间损失（与 B 的 `mixed` 是同一套公式）。
- 另加 `maybe_compute_force_loss`：`SmoothL1` 在 力预测 vs 由 IMU 与物理推力构造的力目标 上，权重来自 `--beta-force`。
- 若 `loss-type` 不是 `mixed`，脚本会 强制改成 `mixed`。

---

## 3. 想用不同 loss 做对比时，可以怎么测（建议）

1. 只改「有没有几何 / 辅助」（在 同一套 lag 结构下）：
    
    - `lag` vs `lag_geo` vs `full`（`geo` 是不带 lag 的纯几何增强对照）。
    - 可调 `--beta_geo`、`--beta_aux`、`--w_rot`、`--w_omega` 做敏感度曲线。
    
2. 只改「时间域 MSE 怎么加权」（plain 类 variant）：
    
    - `baseline`（或 `lag` / `lag_gru`）下：`--loss-type exp` vs `mixed`。
3. 力监督：单独对比 `lag_gru_force` 与 `lag_gru`，并看 wandb 里的 `loss_force`（以及总 loss）。
    
4. 其它训练入口（`train_residual.py`、`train_lstm.py` 等）：目前基本是 固定 `WeightedMSELoss(0.1)`，不算 ablation 里的组合开关。
    
5. 若你要测 「只有反标准化旋转测地、不要 Composite 里那套」，需要 自己接 `DenormRotGeodesicLoss` 或改 `build_criterion`，因为现在主脚本没有这条分支。
    



## 4. 和 variant 的对应关系（方便你列实验表）

train_physres_ablation.pyLines 58-80

def uses_lag(variant):

return variant in {"lag", "lag_gru", "lag_gru_force", "lag_geo", "full"}

def uses_geo_loss(variant):

return variant in {"geo", "lag_geo", "full"}

def uses_aux_supervision(variant):

return variant == "full"

def uses_force_supervision(variant):

return variant == "lag_gru_force"

def uses_plain_temporal_loss(variant):

return variant in {"baseline", "lag", "lag_gru", "lag_gru_force"}

def build_criterion(variant, x_scaler, beta_geo, beta_aux, w_rot, w_omega):

if uses_plain_temporal_loss(variant):

return WeightedMSELoss(lambda_=0.1)

return CompositeAblationLoss(

...

use_geo=uses_geo_loss(variant),

...

use_aux=uses_aux_supervision(variant),

...

)

注意：`geo` / `lag_geo` / `full` 时 `build_criterion` 返回的是 `CompositeAblationLoss`；`baseline`/`lag`/`lag_gru` 返回的是单纯 `WeightedMSELoss`，几何项根本不会进这条训练路径。

如果你说一下最想对比的假设（例如「只要 MSE vs MSE+geo」还是「full 里拿掉 aux」），可以帮你缩成最小的一组命令行矩阵。