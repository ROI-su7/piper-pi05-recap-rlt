# Piper π0.5：RECAP / RLT 真机闭环

面向 AgileX Cobot Magic / Piper 双臂、三路原始相机的复现项目。围绕已经使用的代码链整理，分别提供：

- **RECAP**：数据采集 → LeRobot转换 → π0.5训练/推理 → 人工接管与结果记录 → returns → 回报分布Value训练 → advantage → CFG策略微调 → 部署再采集。
- **RLT**：数据采集 → LeRobot转换 → π0.5训练/推理 → RL Token训练 → 真机推理与人工接管 → replay → actor/critic更新 → 下一回合部署。

这里的Value“分布”指预测回报分布，训练执行使用RLinf。仓库不包含世界模型飞轮、RISE或其他模型的训练方案。保留当前原生JAX π0.5视觉/动作路径、三相机、14维弧度/米和四CAN控制链。

## 目录

```text
common/                 采集、转换、π0.5训练/推理入口；OpenPI本地修改
recap/
  configs/              returns / Value / advantages / CFG
  source/               RECAP采集、数据、策略与测试实现
  patches/              RECAP对固定RLinf版本的必要修改
  run.sh                各阶段独立入口
rlt/
  configs/              RLT真机配置、无额外视觉特征基础配置
  source/               RL Token、actor/critic、replay、接管与RTC实现
  patches/              RLT对固定RLinf版本的必要修改
  run.sh                Stage1 / serve / Stage2 / replay入口
robot/
  cobot_magic/          当前Piper ROS源码与采集转换代码
  follower_hold/        已修正的双从臂带力矩HOLD mux
scripts/                固定版本恢复、安装、检查
sources.lock.json       上游commit、修改顺序、逐文件SHA256
```

OpenPI、RLinf以固定commit作为依赖；`bootstrap.py`下载它们并应用本仓库按功能分离的源码和补丁。重建后的依赖在`deps/`，不上传上游整套框架、数据、权重、环境或编译产物。公开的本地改造代码位于common、recap、rlt和robot，而非依赖原开发机路径。

## 1. 初始化

模型环境Python3.11；ROS Noetic使用系统Python3.8。模型训练与ROS环境分别准备。

```bash
git clone https://github.com/ROI-su7/piper-pi05-recap-rlt.git
cd piper-pi05-recap-rlt
python3.11 scripts/bootstrap.py
bash scripts/install_models.sh
cp configs/local.env.example configs/local.env
```

编辑`configs/local.env`：模型环境解释器、π0.5 JAX checkpoint、匹配norm、示教数据目录、任务配置名。互插用`pi05_connect_strings`；布线用`pi05_wire_all_data_full`；旧LoRA配置保留`pi05_wire`。三者的任务文字和权重/norm应成套选择。

安装器仅新建环境；原有环境可直接设置`PIPER_PYTHON`，不要再次执行安装器覆盖它。`requirements/models.txt`锁定JAX0.5.3、Flax0.10.2、Transformers4.53.2，并使用新环境内的OpenPI Transformers补丁。

已有两个Git源码对象库时可以离线重建，原目录只读：

```bash
python3.11 scripts/bootstrap.py --local-openpi /path/to/openpi --local-rlinf /path/to/RLinf
```

## 2. ROS、相机与数据采集

Piper SDK按硬件/固件版本单独安装到ROS解释器；安装Noetic、catkin、传感器消息、cv_bridge、h5py、OpenCV、dm_env。构建：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -C robot/cobot_magic/Piper_ros_private-ros-noetic
source robot/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
roscore
```

相机使用已安装Astra/RealSense驱动，发布`/camera_f/color/image_raw`、`/camera_l/color/image_raw`、`/camera_r/color/image_raw`。相机SDK及USB序列号属于现场配置，本仓库不携带原机camera_ws的build/devel。

按实际USB-CAN布线设置`can_l_leader/can_r_leader/can_l_follower/can_r_follower`。CAN配置脚本在`robot/cobot_magic/Piper_ros_private-ros-noetic/`；只有全部控制节点停止时才配置接口。

固定主臂示教采集：

```bash
bash robot/launch.sh collect
# 另一个已加载ROS环境的终端：
bash common/pi05.sh collect --dataset_dir "$PWD/data/raw" --task_name wire --max_timesteps 2000
```

固定采集和监督接管为不同ROS profile，只运行其中一个。固定采集使用独立leader的命令输入；人工接管阶段记录的是mux最终仲裁动作。

## 3. 数据转换与π0.5训练

```bash
bash common/pi05.sh convert \
  --datasetDir "$PWD/data/raw/wire" --datasetName local/wire \
  --targetDir "$PWD/data/lerobot/wire" \
  --instruction 'connect the light strings and the battery box to the splitter cable so the lights turn on.'
source common/env.sh
bash common/pi05.sh norm-stats --data.repo-id "$PIPER_DEMO_DATASET"
bash common/pi05.sh train --exp-name wire_round_001 --data.repo-id "$PIPER_DEMO_DATASET"
```

基础权重须先放到配置指向的`deps/openpi/checkpoints/pi05_base/params`，或给原OpenPI命令传匹配的weight-loader参数。沿用原训练好的任务模型时直接配置它的checkpoint和norm，无需重训。

原始示教保持原生帧率，不能简单下采样后假定动作速度不变。图像为三路RGB，state/action为左臂6关节+夹爪、右臂6关节+夹爪，共14维；模型内部padding为32。

## 4. π0.5推理与人工接管

启动监督接管profile：

```bash
bash robot/launch.sh takeover
bash recap/native_server.sh
# 独立终端，键盘和结果记录由它统一拥有：
bash recap/run.sh collect --profile takeover
# 独立终端，策略客户端：
bash recap/client.sh takeover
```

RECAP：`I`推理/开始或恢复回合，`T`接管，`C`成功，`X`失败，空格HOLD，`H`HOME，`Q`退出。接管须等待确认ACTIVE；按I恢复时先回HOLD再请求INFERENCE。不要同时运行另一个键盘控制器。采集按15Hz，实际命令插补发布200Hz。

双从臂纯推理使用`bash robot/launch.sh followers`和`bash recap/client.sh followers`。默认入口明确使用`robot/follower_hold`，避免原Cobot follower mux在HOLD/反馈间隙失能。详细协议、角色握手和复查结果见[人工接管核对](docs/TAKEOVER_AUDIT.md)。

## 5. RECAP闭环

每轮使用新的rollout快照和输出目录；先在local.env设置`RECAP_DATASET`、`RECAP_RUN_DIR`。RECAP的returns会写数据集meta，因此示教必须复制到`RECAP_SFT_DATASET`，不能指向唯一原始数据。

```bash
source common/env.sh
# 配置中的示教副本默认 data/recap/demos_copy，首次建立只做一次：
python3.11 scripts/copy_dataset.py "$PIPER_DEMO_DATASET" "$PWD/data/recap/demos_copy"
bash recap/run.sh export
bash recap/run.sh convert
# 专用训练Ray head，先source本轮环境，再启动：
source deps/rlinf/examples/recap_piper/env.sh
ray start --head --port=6385
bash recap/run.sh returns
bash recap/run.sh value
# 设置本轮实际Value checkpoint：
export RECAP_VALUE_CHECKPOINT=/absolute/path/to/this_round/value_checkpoint
bash recap/run.sh advantages
bash recap/run.sh cfg
bash recap/run.sh serve --checkpoint /absolute/path/to/global_step_N/actor --guidance-scale 1.5
```

serve替代本轮原生服务器，仍使用6016端口和同一个接管客户端。下一轮采集成功/失败/人工纠正，再导出新快照、更新Value与策略。配置默认融合示教与rollout；示教与人工纠正的positive标记遵循当前RECAP实现。

继承的当前配置还保留可选接管惩罚`data.takeover_penalty: 300`，这是项目扩展；只使用原RECAP奖励时运行returns传`data.takeover_penalty=0`，后续advantage使用该次生成的returns。详细步骤与作用见[RECAP说明](recap/README.md)。

## 6. RLT闭环

RLT从同一原生π0.5提取三相机prefix，训练RL Token；然后冻结π0.5与token，用轻量actor/critic在真实经验及人工纠正replay上更新。无需JAX→Torch转换来执行基础动作。

```bash
# 已配置PIPER_JAX_CHECKPOINT、PIPER_DEMO_DATASET：
bash rlt/run.sh stage1 --steps 2000
# 设置本轮生成的有效格式token，再启动机器人本地服务：
export RLINF_STAGE1_RLT_TOKEN=/absolute/path/to/step_2000/rlt_token.pt
bash rlt/run.sh serve
# 启动roscore、三相机、四CAN takeover profile后：
source common/env.sh
export RLINF_STAGE1_RLT_TOKEN=/absolute/path/to/step_2000/rlt_token.pt
export RLINF_JAX_REF_HOST=127.0.0.1 RLINF_JAX_REF_PORT=6006
export RLINF_NODE_RANK=0
ray start --head --port=6379
bash rlt/run.sh preflight
bash rlt/run.sh stage2-base
```

`stage2-base`保留RLT、RTC、接管和归一化delta动作，关闭额外head-pose特征；`stage2`保留原任务的head-pose特征版本，需设置`PIPER_HEADPOSE_WEIGHTS`。两者检查点与replay维度分别匹配。

RLT键盘由EnvWorker统一管理：`I`释放推理，`B`进入关键窗口/开始记录，`T`人工接管，`I`恢复，`C/X`标结果，空格HOLD，`H`HOME。不要启动RECAP记录器或OpenPI普通客户端去争用同一个policy topic。温启动期间执行基础策略，达到配置中的replay与学习更新数才启用actor。

后续复训或续训：

```bash
export RLINF_REPLAY_LOAD_PATH=/absolute/path/to/replay/rank_0
PIPER_RLT_CONFIG=piper_rlt_base bash rlt/run.sh replay
# 恢复整个Stage2时：
bash rlt/run.sh stage2-base runner.resume_dir=/absolute/path/to/checkpoints/global_step_N
```

使用同一个token指纹、动作归一化和特征维度。人工接管按实际executed action记录；当前配置保留JAX reference用于人工BC目标，并在接管边界按配置终结机器片段。见[RLT说明](rlt/README.md)。

## 7. 检查与验证范围

```bash
source common/env.sh
bash scripts/check.sh
bash recap/run.sh cfg --dry-run
bash rlt/run.sh stage2-base --cfg job --resolve
```

已从固定Git对象离线重建上游并应用全部源码修改；已进行接管状态机、CAN发送屏障、原生HOLD旧动作丢弃、实际数据转换、RECAP标签、RLT replay/人工纠正/RTC和配置检查。具体数量与限制记录在[验证报告](docs/VALIDATION.md)。本次整理没有重新启动真机或完成一次全量GPU训练，新环境安装仍须在目标机器执行。

## 来源与许可

[OpenPI](https://github.com/Physical-Intelligence/openpi) · [RLinf](https://github.com/RLinf/RLinf) · [RECAP论文](https://arxiv.org/abs/2511.14759) · [RLT研究说明](https://www.pi.website/research/rlt)。版本及修改SHA见sources.lock.json；Piper原始MIT声明与Apache声明保留在licenses/及原文件中。

权重、演示录像、replay、API凭据及运行环境均不随源码上传。详细归属见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
