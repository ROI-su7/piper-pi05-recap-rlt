# 验证记录

日期2026-11-17。本次整理保留既有执行链，没有启动ROS节点、相机、CAN或机械臂。

## 已完成

- 从本机只读Git对象恢复OpenPI、RLinf固定commit，再应用全部已发布补丁/源码。上游RLinf commit已通过GitHub API确认存在。
- 200项CPU检查通过，另有4个subtest通过，无跳过。包括原Piper四CAN profile、leader角色握手、CAN TX guard、HOLD/HOME边界、原生客户端在途结果丢弃与ACK排空，以及RECAP导出/returns和RLT人工纠正、replay、RTC。
- RECAP CFG和RLT基础版Stage2配置均可由便携入口展开；RECAP配置包含示教副本与自主/人工纠正rollout两份数据。
- SHA256校验覆盖128个已发布的Piper补丁/源码操作；依赖和算法分开维护。硬件源码的license原样保留。

## 未在本次重新执行

- 新机器完整依赖安装、GPU全量SFT/Value/CFG/Token/actor训练。
- 本次发布仓库的新真机接管、运动和任务成功率验收。
- 外部权重与实际数据的下载。它们由使用者配置，未上传GitHub。

scripts/check.sh可复现上述CPU检查；模型环境需bootstrap后准备好。模型/学习阶段入口指向当前源码实现，配置展开不会启动训练或硬件。
