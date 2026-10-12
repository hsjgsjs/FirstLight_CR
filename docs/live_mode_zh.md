# 实战模式：让 FirstLight 在 MuMu 上和你打

[首页](../README.zh-CN.md)

目标：你在手机上用自己的 Null's 账号操作，MuMu 上的另一个账号由 FirstLight 模型控制，两边打友谊战。

做法：probe 的实战模式挂到游戏自己运行的对局上，在游戏线程里采集 FirstLight 的完整观测；BattleEnvV1 和 V4 策略按 FirstLight 自己人机对战（`offline_agent`）的方式做决策；出牌变成 adb 在 MuMu 屏幕上的点击。挂接方式、控制器偏移、评分弹窗修复、点击时序规则都来自 [Jason-XII/clash-royale-battle-engine](https://github.com/Jason-XII/clash-royale-battle-engine)，它和 FirstLight 用的是同一个 libg 构建。

**只用于你自己的账号。**

## 准备（每次更新代码后）

1. 拉取最新代码，用 `build_offline_engine.ps1` 和 `install_offline_engine.ps1` 重新构建并安装 APK（probe 有改动）。
2. MuMu 渲染模式用 Vulkan。
3. 让游戏联网启动并转发控制端口：
   ```powershell
   .\.venv\Scripts\python.exe -m native_runner.live_diagnose start
   ```
4. 用完后恢复断网：`.\.venv\Scripts\python.exe -m native_runner.live_diagnose offline`（下次运行 `start_offline.ps1` 时也会自动恢复）。

## 第 1 步：校准点击位置

点击坐标取决于 MuMu 的分辨率。第一次运行前先校准：

1. 手机上向 MuMu 账号发起友谊战，进入对局后运行：
   ```powershell
   .\.venv\Scripts\python.exe -m native_runner.live_play shot --owner 1
   ```
   `--owner` 是 MuMu 账号在这局里的座位；不知道的话先不加，它会列出双方的 accountId 和牌组。
2. 它会保存两张图到 `runs\live\`：
   - `shot-<时间>.png`：原始截图；
   - `shot-<时间>-layout.png`：叠加了当前布局的截图。青色网格是 18×32 的格子，橙色圆圈是按布局算出的塔的位置，紫色圆圈是 4 个手牌位置，黄色圆圈是英雄技能按钮。
3. 如果橙色圆圈正好压在塔上、紫色圆圈压在 4 张手牌上，就说明布局是准的。在 `runs\live\layout.json` 里把 `"calibrated"` 改成 `true`。
   如果对不上，把这两张图发给我，我来改 `layout.json`。

## 第 2 步：让模型出牌

```powershell
.\.venv\Scripts\python.exe -m native_runner.live_play run --account-id <MuMu 账号的 accountId>
```

- 运行后它会一直等待。你每发起一局友谊战，它就挂上去打这一局，打完继续等下一局。按 Ctrl+C 停止。
- `--account-id` 只需要第一次给，之后记在 `runs\live\seat.json` 里。第 2 轮样本里，owner 1（accountId 71383344）的圣水一直是满的，没人操作，这应该就是 MuMu 账号。
- `--dry-run`：只做决策、写日志，不点击。适合先看模型会怎么出牌。
- `--games N`：打 N 局后退出。
- `--device cpu|cuda`：默认有显卡就用 cuda。
- 每局的日志写在 `runs\live\<时间>.jsonl`，内容包括每次决策（推理耗时 `inference_ms`、游戏线程上的采集耗时 `capture_us`）、每次点击、被跳过的点击和原因、疑似丢失的点击。打完把日志发给我。

### 点击规则（来自 Jason 在 MuMu 上的实测）

- tick 94 以前点击无效，所以这之前的出牌一律跳过。
- 出牌是两次点击：先点手牌，再点格子。牌在点击后约 21～25 tick 生效，这时下一张牌才发到这个槽位。在这之前，这个槽位在屏幕上是空的。
- 新发的牌要过约 100ms（2 tick）才接受点击。
- 所以对同一个槽位：上一张还没生效、或者新牌刚发下来，这时的点击都会被跳过，并记进日志。BattleEnvV1 自己也会把在途的牌从可选项里去掉，并预留它的圣水。

## probe 命令

| 命令 | 作用 |
| --- | --- |
| `live on` / `live off` / `live status` | 开关实战模式；`status` 返回是否已挂上对局、tick、采集次数、上次采集在游戏线程上花了多少微秒（`captureMicros`） |
| `live observe-atomic [c p v r]` | 在游戏线程上、对局刚更新完的那一刻采集一次完整观测。可选的 4 个数分别是战斗事件、相位、可见性、剩余时间这四个事件环的游标，带上后只发送游标之后的事件 |

实战模式开启时，普通的 `observe`、`observe-rich`、`observe-atomic` 会被拒绝，因为在控制线程上读一个正在运行的对局，读到的状态前后不一致。

## 诊断记录

### 第 1 轮（2026-10-12）

- 5 个样本全部通过 FirstLight 的 atomic/rich 校验，General 模型能在实战数据上正常决策。
- 发现并修复：挂接时只登记了战斗事件，相位、移动、角色状态、塔兵四类遥测一直报“不完整”。

### 第 2 轮（2026-10-12）

- 5 个样本全部通过校验。8 类遥测全部 `complete=true`，上一轮的修复生效了。
- 发现：probe 每次都会把事件环里保留的全部历史发出来。开局 15 秒后一次采集就从 200KB 涨到 1.7MB，游戏线程上的耗时从 62ms 涨到 172ms。对局越往后越大，最多到几 MB，会让游戏卡顿。
- 修复：事件环改为只发增量。BattleEnvV1 本来就按序号逐条消费事件，并记着每个环消费到了哪一条，所以每次采集都带上它的游标，probe 只发游标之后的事件。两轮样本、两个座位共 4 组离线回放，增量模式和整环模式给出的决策完全一致。
- 离线回放把录好的样本当成一局实战来放，走的是和 `live_play run` 相同的代码（`env.step`、出牌回执校验、在途核对），只是不点击：
  ```powershell
  .\.venv\Scripts\python.exe -m native_runner.live_diagnose replay runs\live-diagnose\<时间> --owner 1
  ```
  第 2 轮 owner 1：tick 666 下骑士到 [3,18]，tick 727 下弓箭手到 [0,25]，tick 914 下一张费用为 1 的牌到 [8,31]。录像里这些牌当然没有真的下出去，所以日志里会记为疑似丢失。

### 已知限制

- 实战采集里没有牌组的进化/英雄槽位标记。现在按天梯规则处理：只有前两个槽位算进化槽，万能槽和英雄槽暂时不识别。
- 英雄技能按钮的位置还没校准。按钮只在英雄在场时出现，所以要在英雄在场时截图（`shot`）。
- 点击 → 生效这段延迟训练里没有，模型是按“下一 tick 就生效”训练的。Jason 的经验是，这会让模型防守慢半拍。
