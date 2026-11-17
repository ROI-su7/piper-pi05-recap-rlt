# RECAP：经验与人工纠正更新π0.5

独立实现位于source/rlinf/rlinf/recap_piper；训练配置位于configs；框架改动在patches/rlinf.patch。执行 `bash recap/run.sh <stage>`，不会调用RLT actor或RL Token。

1. 使用原生JAX π0.5与四CAN接管采集成功、失败和人工纠正回合。记录mux最终命令、policy命令、leader命令及intervention，C/X给回合标结果。
2. export构建新的LeRobot快照；原350条示教单独复制。两份都参与returns、Value、advantage和CFG。
3. returns按照当前项目逐步奖励与终止结果构造回报，Value使用当前RLinf回报分布实现。可选接管惩罚是项目扩展，关闭时returns与advantages须保持一致。
4. 人工纠正按配置强制为positive；自主动作由n步奖励、后继Value与当前Value得到优势并转二分类。CFG训练保留无条件样本。
5. serve加载本轮动作专家适配器，positive条件/CFG生成动作；复用原生客户端的RTC、HOLD、T/I握手再次采集，进入下一轮。

完整命令见根README第5节。训练输出与数据放在data/recap、artifacts/recap、runs/recap；新一轮更换RECAP_DATASET与RECAP_RUN_DIR，不覆盖前一轮快照。对照源检查点的任务文字、norm和相机顺序必须相同。

共享Piper ROS层独立于算法。禁止同时运行两个policy客户端/键盘；RECAP用6016和独立Ray6385。这里沿用已经使用的分布式训练代码并锁定上游版本，没有以另一个简化Value代替当前实现。
