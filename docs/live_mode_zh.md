# 实战模式诊断（第 1 步）

[首页](../README.zh-CN.md)

目标：你在手机上用自己的 Null's 账号原生操作，MuMu 上的另一个账号由 FirstLight 模型控制，两边打好友战或训练营练习。

这一步只做一件事：**确认 FirstLight 的完整观测能不能在联网实战对局里读出来，并通过 FirstLight 的校验。** 这一版只读，不会出牌。

实战挂接的做法，以及控制器偏移、评分弹窗修复、带标记指针修复，都来自 [Jason-XII/clash-royale-battle-engine](https://github.com/Jason-XII/clash-royale-battle-engine) 的实战模式，它和 FirstLight 用的是同一个 libg 构建。

**只用于你自己的账号。**

## 步骤

1. 拉取最新代码，用 `build_offline_engine.ps1` 和 `install_offline_engine.ps1` 重新构建并安装离线 APK。这一步更新的是 probe。
2. 确认 MuMu 实例的渲染模式是 Vulkan，然后运行：
   ```powershell
   .\.venv\Scripts\python.exe -m native_runner.live_diagnose start
   ```
   它会做三件事：解除这个实例的断网规则、以联网方式重启游戏、转发控制端口。进入大厅后，确认账号正常登录。
3. 运行：
   ```powershell
   .\.venv\Scripts\python.exe -m native_runner.live_diagnose run
   ```
   然后在游戏里开一局训练营，或者和你手机上的账号打一局友谊战。脚本挂上对局后，会每隔 3 秒采集一次完整观测，共采 5 次，每次都跑 FirstLight 的校验。
4. 把脚本最后打印的文件夹 `runs/live-diagnose/<时间>/` 整个发给我。里面有：
   - `summary.json`：每个样本的校验结果和概要；
   - `sample-*.json.gz`：原始采集数据。我会在离线环境里，用它们把完整的观测流程再跑一遍。
5. 用完后恢复断网：
   ```powershell
   .\.venv\Scripts\python.exe -m native_runner.live_diagnose offline
   ```
   下次运行 `start_offline.ps1` 时，它也会自动恢复断网。

## 新增的 probe 命令

| 命令 | 作用 |
| --- | --- |
| `live on` / `live off` / `live status` | 开关实战模式，查看是否已挂上对局、当前 tick、已采集次数 |
| `live observe-atomic` | 在游戏线程上，于对局刚更新完的那一刻采集一次完整观测（普通观测 + rich 观测） |

实战模式开启时，普通的 `observe`、`observe-rich`、`observe-atomic` 命令会被拒绝，因为在控制线程上读一个正在运行的对局，读到的状态前后不一致。

## 第 1 轮结果（2026-10-12）

- 5 个样本全部通过 FirstLight 的 atomic/rich 校验；双方牌组、手牌、循环、圣水都能读到；游戏线程上每次采集 31～47ms。
- 发现并修复：挂上实战局时只登记了战斗事件，相位、移动、角色状态、塔兵这几类遥测没有登记，所以一直报“不完整”。现在挂接时统一登记全部遥测。
- 用保存的样本离线跑通了 `BattleEnvV1` → 张量化 → General 模型，模型能正常给出决策：
  ```powershell
  .\.venv\Scripts\python.exe -m native_runner.live_diagnose replay runs\live-diagnose\<时间>
  ```
- 已知差异：实战采集里没有牌组的进化/英雄槽位标记。现在按天梯规则处理：只有前两个槽位算进化槽，万能槽和英雄槽暂时不识别。
