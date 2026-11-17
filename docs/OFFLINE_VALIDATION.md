# 真实模型端到端离线验证

本项目提供可执行入口 `scripts/offline/run.py`。它读取已有真实录像，经过正式记录/转换接口，运行真实原生JAX π0.5训练、RECAP训练、完整RLT Token和actor/critic训练，最后严格重载检查点推理。

## 命令

先bootstrap、准备模型环境和本地资产；权重不得指向随机初始化替身。

```bash
source common/env.sh
"$PIPER_PYTHON" scripts/offline/run.py \
  --demo "$PIPER_DEMO_DATASET" \
  --rollout /absolute/path/to/lerobot_v5 \
  --checkpoint "$PIPER_JAX_CHECKPOINT" \
  --siglip /absolute/path/to/siglip2-so400m-patch14-224 \
  --gemma /absolute/path/to/gemma-3-270m \
  --output "$PWD/runs/offline_4090_new"
```

中断后使用同样参数加`--resume`。输出目录首次须不存在。默认当前任务为pi05_connect_strings，任务文字与模型/norm必须匹配；本入口是小步执行验证，不能替代完整任务训练。每个学习阶段默认2步，RLT的actor/critic更新频率遵循该配置。

## 实际执行顺序

1. 从真实示教及rollout截取64帧尾段，保留原始位置、结果、接管标记；实际调用EpisodeWriter与Cobot HDF5/LeRobot转换接口。PNG逐帧比对RGB，状态和动作容差1e-6。
2. 在原生JAX完整π0.5基座上训练动作专家LoRA。保存新JAX检查点，重载推理。后续RECAP和RLT都从这份新SFT产物开始。
3. 用该模型提取真实768×2048视觉prefix缓存，然后在独立进程训练完整745,715,712参数RLT Token。缓存校验模型路径、配置名、来源标记和形状；不是随机特征。24GB配置使用CPU参数/Adam卸载。
4. 通过正式JAX→Torch转换器保留原生LoRA分支。RECAP计算真实回报，训练回报分布Value，为全部帧计算优势，微调专家LoRA；重载并执行普通和CFG1.5推理。
5. 重载同一新JAX策略与新Token，使用生产CompactRLTPolicy得到真实观测的2048维z_rl及reference。保存真实记录动作与接管归属，构成RLT replay。
6. 用生产RLT replay-only actor/critic学习器实际更新，保存完整检查点；读取DCP optimizer step证明更新发生，再严格重载并执行生产predict_action_batch。
7. 校验所有阶段报告、标签覆盖、人工纠正positive标签、optimizer有限值与最终动作输出。输出validation.json。

离线replay没有实时RTC队列，bridge_delta明确为0；末端动作按真实有效长度mask，模型长度结束不会伪造执行动作。真实RTC/HOLD发布屏障和T/I恢复另由CPU回归测试覆盖。

## 2026-11-17 本机4090结果

| 阶段 | 结果 |
|---|---|
| 数据 | 7段×64帧=448帧，含37人工接管帧；RGB/状态/动作往返通过 |
| 原生π0.5 SFT | 2步专家LoRA优化，保存JAX检查点并重载输出[50,14] |
| RECAP Value | 2步真实回报分布Value优化，SigLIP2/Gemma预训练骨干 |
| RECAP标签 | 448条：256正、192负；37人工帧全部positive |
| RECAP策略 | 2步真实LoRA优化，严格重载普通/CFG推理均为有限[50,14] |
| RLT Token | 完整7.46亿参数，2步loss：3.465929→3.189227 |
| RLT特征/replay | 新Token指纹1db6126a8156f3d7；35条转移，315个有效记录动作 |
| RLT学习 | 2轮；实际actor optimizer step=1、critic step=2，状态有限 |
| RLT重载推理 | 严格载入actor，生产接口输出有限[1,10,14] |

37个接管观察帧中最后一个没有后继观测，RLT因此监督36个人工执行动作；其余有效前缀同样按真实长度计数。

本机RTX4090 24GB、32GB主机RAM，核心依赖版本记录在validated_versions_4090.json。Token与JAX同驻训练曾超显存，现通过分进程缓存完整原生特征和CPU卸载解决。验证过程中未启动ROS/CAN或机械臂；不宣称小步训练已收敛或取得真机收益。

## 产物和修复

`validation.json`是总报告；`data/data_report.json`是源回合/裁剪/接管来源；各阶段.log记录真实执行。本次完整产物约42GB。模型产物、图像、数据和缓存只保存在忽略的runs目录，不上传GitHub。公开摘要在docs/offline_4090_20261117.json。

实际执行发现并修复：PNG模式图像被错误声明为video；示教优势计算没有继承全局Piper相机类型；单样本Value方差日志的NaN；24GB下双框架同驻内存冲突。对新增数据契约和依赖刷新行为有回归测试。

拉取新版本后，如果原deps版本标记过期，可用`python3.11 scripts/bootstrap.py --refresh`重建；只处理带生成标记的依赖，并保留.previous_备份，不删除数据或权重。
