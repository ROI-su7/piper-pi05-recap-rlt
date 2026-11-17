# 人工接管代码核对

日期2026-11-17。依据当前cobot_magic、原生OpenPI客户端和RLinf Piper路径核对；ROS/CAN代码保留原实现，新增的是便携入口、包路径检查与测试。

## 1. 控制归属

每个物理CAN只有一个driver/SDK owner；policy只发布/teleop/policy/joint_left/right。four-state mux或固定follower mux是/teleop/command/joint_left/right的唯一发布者。从臂driver只消费仲裁目标，实际执行命令还在/master/joint_left/right镜像供记录器读取。

固定示教采集使用独立read-only fixed-master reader、采集mux和从臂driver，动态接管使用four-state mux及leader role driver，两种profile互斥。

## 2. T请求与确认

four-state mux先进入带力矩HOLD，记录请求时间；必须收到本次请求之后的policy_hold_ack，并等待policy发布静默与主从姿态对齐，再关闭leader OUTPUT TX门。leader driver在发送锁屏障后回报tx_quiet；静默间隔满足后才切换主臂角色。等待新鲜leader反馈及两个主臂的MASTER/MASTER_ARMED状态后标记TAKEOVER/ACTIVE。

任何一项未满足，保持HOLD或按原状态机超时收敛，不能把请求T等同于接管已完成。记录器只在确认的运行阶段记录人工intervention。

## 3. 实际角色路径与辅助函数

当前运行循环以MASTER/OUTPUT角色握手执行动态接管。运行时_enter_master_role的allow_disable_fallback=false，失败回到OUTPUT HOLD，不走固定采集准备时允许的禁用回退。

leader源码也包含_enter_drag_teach辅助函数及其单测；当前run循环没有调用它。不能仅根据函数存在或旧说明文字就把默认接管路径解释成DRAG_TEACH。此版本未重写底层角色协议或选择另一条固件路径。

## 4. HOLD与恢复

原生OpenPI环境先更新HOLD generation，在_command_lock屏障排空最后一次ROS发布后清空插补与平滑缓存，再发布policy_hold_ack=true。HoldAwarePolicy在归属epoch变化时reset broker并读取新观测；旧epoch在途结果被丢弃，Environment发布前也复查generation。

RLT隔离executor具有相同的发布排空与ACK语义；恢复时显式reset，参考窗口、执行offset和RTC bridge一起对齐。人工接管与策略恢复之间不重用旧action chunk。

## 5. 原Cobot follower mux的区别

原Piper_ros工作区的piper_policy_follower_mux是保留的对照源码。当前入口followers改用robot/follower_hold/的powered-HOLD版本，其实现marker为lingbot_piper/piper_policy_follower_mux powered-hold v1。启动前校验rospack路径和marker；记录器拒绝未标记的原follower mux。

监督四臂takeover使用four-state mux，不需要follower-only enable mux。不可同时启动followers和takeover profile。

## 6. 数据与两套学习语义

RECAP采集保存mux最终action、policy_action、leader_action和intervention；失败和成功回合均保留。人工纠正按当前优势配置标positive，仍读取完整回合结果作为Value监督。

RLT使用实际执行前缀及明确归属标签，不能把请求到T的等待时间标为人工动作。当前配置human_rows_keep_jax_reference=true，人工动作仍为监督目标，JAX参考保留在输入用于学习纠正；takeover_as_failure按配置在最后机器步终结机器片段，并非把整条人工救回的回合伪造成失败。

## 7. HOME与退出

HOME使用原launch记录的姿态，结束为HOLD。当前监督launch默认先将从臂夹爪打开0.06m并等待1.5s再HOME，属于现有行为；若现场需要原不释放物料的行为，使用home_release_gripper_m:=0。

Q/软件HOLD不等于物理急停。客户端/记录器退出不应把缓存动作继续发布；不要另开一个键盘或policy客户端竞争控制归属。

## 8. 离线核验范围

覆盖原hardware role/CAN发送屏障/collect profile测试，RECAP仲裁动作与intervention导出，原生HOLD ACK时序和旧推理结果丢弃，RLT executor ACK、人工reference/BC、takeover边界、RTC执行窗口和replay恢复。

测试使用fake ROS/SDK；测试进程禁止创建SocketCAN。当前整理没有连接真机，不将离线状态机通过描述成新的真机验收。
