# RLT：冻结π0.5与Token，更新actor/critic

独立实现位于source/rlinf/rlinf/algorithms/rlt、models/embodiment及envs/realworld/piper；配置在configs，框架改动在patches/rlinf.patch。执行 `bash rlt/run.sh <stage>`。

1. 采集和转换真实三相机示教，按任务配置训练/复用原生JAX π0.5。
2. Stage1从冻结原生JAX视觉prefix训练Token encoder/decoder，三相机768图像token；格式、模型配置和encoder指纹写入检查点。
3. 本机compact服务执行JAX π0.5并输出紧凑z_rl，跨机传输可选；基础动作不经过另一个视觉模型实现。
4. Stage2读取reference、proprio、z_rl，归一化delta空间actor/critic更新；记录实际executed action、reference、接管归属与有效前缀。
5. T接管由四CAN mux控制，人工BC/replay遵循当前配置；I恢复清空旧动作与RTC状态。replay、优化器和调度状态可继续保存/恢复。

根README第6节给出命令。stage2-base为纯RLT特征配置；stage2保留原任务额外head-pose特征，权重自备。任何重载都要匹配token指纹、action_norm_stats、特征维度和任务文字。

在线只运行RLT EnvWorker作为policy发布者，不另开普通OpenPI客户端或RECAP记录器。服务6006，训练Ray6379。普通推理、温启动、actor接管和人工纠正分别有明确模式标记；学习使用实际执行窗口与执行偏移，保留当前RTC连续性处理。
