# 手机游玩与精英（英雄）问题分析

[首页](../README.zh-CN.md) · [安装指南](getting_started.md)

这份文档回答两个问题：

1. 能不能脱离 PC 上的 MuMu，直接在手机上玩？
2. 精英（英雄形态）为什么渲染异常、为什么放不出技能？

结论先放在前面：

- **“手机显示和操作，PC 跑模型”可以做，这次已经在代码里实现。** 手机通过 USB 或无线调试连到 PC 后，整局都在手机上操作：拖牌、点牌出牌，点自己场上的英雄或冠军放技能。PC 只负责开局和运行 AI。
- **“完全不要 PC”目前做不到。** 原因见第 1.3 节。
- **英雄技能放不出来**：找到并修复了两个代码层面的原因（第 2.1 节）。另外，原来直接在游戏画面上点击的方式本来就没有技能操作，这次补上了“点英雄放技能”。
- **英雄模型不显示**：原因已经确认。英雄本体用的 3D prefab 和动画状态机 `.json` 文件，原版游戏只在需要时联网下载，离线引擎拿不到。现在可以用 `update_assets.py` 把这些文件补进离线实例（第 2.3 节）。

> 以下改动都只通过了主机侧的编译检查和单元测试（第 4 节），**还没有在真机上验证过**。第 1.2 节列出的风险需要你在手机上实际测一遍。

---

## 1. 手机游玩

### 1.1 项目原来的结构

```
PC（Windows）                                  MuMu（Android 12, root）
┌───────────────────────────┐                 ┌──────────────────────────────┐
│ user_interface.py 控制台   │                 │ 改过的 Null's Royale APK      │
│ offline_agent / 模型推理    │ ── TCP 26789 ──▶│  libcrprobe.so（probe）        │
│ native_overlay.py 透明点击层│   adb forward   │   ├ 控制端口 127.0.0.1:26789   │
└───────────────────────────┘                 │   ├ 观战（回放）模式渲染对局      │
                                              │   └ 钩住触摸 GameApp.nOnTouchEvent│
                                              └──────────────────────────────┘
```

- 对局使用游戏自带的**回放/观战渲染**（native-render）。双方的出牌都通过控制端口注入命令，这就是“纯接口下兵 + 观战模式”。
- probe 里原本就有一个**触摸拦截器**（`native_touch_interceptor.cpp`）：在游戏画面上点手牌、再点场地，就能直接出牌。但它有三个限制：
  - 坐标写死为 1080×1920；
  - 双方手牌都能点；
  - 没有放技能的操作。
- 在 MuMu 上，`native_overlay.py` 在窗口上盖了一层透明点击层，技能按钮放在侧边面板里。

### 1.2 方案 A：手机显示和操作，PC 当大脑（已实现）

在新的结构里，手机取代 MuMu：probe 运行在手机上的游戏进程里，PC 通过 `adb forward` 连接手机的 26789 端口。PC 端的控制台、模型、对局逻辑都不需要改。

这次的改动：

| 位置 | 改动 |
| --- | --- |
| `probe/native_touch_interceptor.cpp` | 触摸布局改为运行时可配置（`touch layout ...`）；新增“只允许控制某一方”（`touch owner N`），在人机对战中 AI 那一方的手牌点不动；手牌可以**直接拖到场地**出牌（原来的“先点牌再点场地”仍然可用）；没选牌时**点自己场上已就绪的英雄/冠军**就会释放技能（范围 2.5 格内，取最近的一个）；校准模式（`touch calibrate on`）期间只记录点击坐标，不触发任何操作。 |
| `probe/native_touch_probe_glue.inc` | 实现上述命令，并扩展 `touch status` 的输出（布局、最近一次按下和抬起的坐标、技能点击计数）。 |
| `native_runner/phone_touch.py` | 手机工具：`connect` 负责 adb 端口转发并启动游戏；`calibrate` 在手机上点 7 个位置，自动算出布局并保存；`apply` 和 `status` 分别用于应用设置和查看状态。 |
| `user_interface.py` | 在 `.env` 里设置 `CR_DEVICE_MODE=phone` 后：开局时通过 adb 连接手机，而不是启动 MuMu；不再弹出 Windows 透明层，改为把校准好的布局和“人类控制哪一方”发给手机。 |

#### 使用步骤

1. 准备一台 **arm64** 手机，开启 USB 调试，或者 Android 11 及以上的无线调试。
2. 安装离线 APK，并保留游戏已经下载好的更新资源（见下面“风险 1”）。
3. 在 `.env` 里填写：
   ```
   CR_DEVICE_MODE=phone
   CR_ADB=C:\path\to\adb.exe
   CR_PHONE_SERIAL=<adb devices 里显示的序列号，或无线调试的 ip:port>
   ```
4. 连接手机，然后在控制台里正常开一局（手动对局或人机对战都可以）。
5. **第一次使用需要校准**。对局画面出现在手机上之后，在 PC 上运行：
   ```powershell
   .\.venv\Scripts\python.exe -m native_runner.phone_touch calibrate
   ```
   按终端提示在手机上依次点 7 个位置：下方第 1 张和第 4 张手牌、上方任意一张手牌、左桥、右桥、上方左侧公主塔、下方右侧公主塔。结果保存在 `runs/interface/phone_touch_layout.json`，以后每次开局会自动应用。
6. 对局中的操作：
   - **出牌**：拖动手牌到场地，或者点手牌再点场地。
   - **放技能**：不选牌，直接点自己场上的英雄或冠军。

#### 真机风险（MuMu 上不存在，需要你实测）

1. **安装时如何保留更新资源（最大的障碍）。**
   - 离线 APK 换了签名，替换安装必须先卸载原版，而卸载会清空 `/data/data/nullsroyale.rel.free/update`，也就是游戏内下载的更新。
   - `install_offline_engine.ps1` 之所以要求 root，就是为了先备份再恢复这份数据。
   - 手机有 root：直接沿用现有的 `install_offline_engine.ps1` 流程，`.env` 里的 serial 填手机的。
   - 手机没有 root，有两条可以尝试的路，**都没有实现，也没有验证**：
     - 让构建脚本给 APK 加上 `android:debuggable="true"`，安装后用 `adb shell run-as nullsroyale.rel.free` 把从 MuMu 里导出的 `update/` 目录写回去；
     - 试一下 `pm uninstall -k`（卸载但保留数据）后再安装。不确定系统是否允许签名不同的包接管保留下来的数据。
2. **inline hook 在真机上能否生效。**
   - probe 会用 `mprotect` 把 libg 的代码页临时设为可写可执行，然后打补丁。
   - MuMu 是 x86 加 ARM 转译，真机是原生 arm64，并且有 SELinux 和 W^X 限制，这一步可能被拒绝。
   - 判断方法：`adb logcat -s CRNativeTouch CRProbe`。如果出现 `mprotect ... failed` 或 `prologue mismatch`，就是这一步出了问题。
3. **断网。**
   - 原流程在 MuMu 上用 iptables 把游戏隔离在离线状态，没有 root 的手机做不到。
   - 可以改用系统的“应用联网权限”开关禁止它联网，或者开飞行模式并通过 USB 调试连接。
   - 控制端口走本机回环，不受这些设置影响。
4. **屏幕必须常亮，游戏必须在前台。** 渲染停止后对局也会停。
5. **校准布局与机型绑定。** 换手机，或者改了分辨率、显示缩放之后，要重新校准。可以用 `phone_touch reset` 恢复 MuMu 的默认布局。

### 1.3 方案 B / C：完全脱离 PC（未实现）

- **方案 B：在手机上用 Termux 跑整套 Python。**
  - 模型约 48 MB（大约 1200 万参数），算力上手机 CPU 能跑。
  - 障碍：
    - 需要在 Termux 里装 Python 3.12 和 torch 2.5；
    - 控制台是 Tkinter + PowerShell，要另外写一个命令行或网页启动器；
    - 张量化和观测校验部分的 Python 代码很重，手机上每 5 个 tick（250 ms）完成一次决策会比较吃力。
  - 方案 A 跑稳之后可以考虑做这一步：PC 端的逻辑基本可以原样搬过去，只是把 `adb forward` 换成本机回环连接。
- **方案 C：把模型放进 probe（C++/ONNX）。** 需要把整个 tensorizer 和观测管线重写成 C++，工作量最大，不建议。

---

## 2. 精英（英雄形态）问题

控制台里的“精英（英雄）”就是 Hero 形态，形态掩码为 2，对应 `class 203` 的卡牌。

### 2.1 技能放不出来：找到的问题和修复

1. **一个英雄控制器“不在场”，会把同一方的整个技能状态隐藏掉。**
   - 英雄形态的角色数据是派生出来的 EXT 记录，它在 `+0x40` 处的全局 ID 是 0。
   - 原来的 `champion_controller_is_exact_absent` 要求这个 ID 不为 0，所以英雄阵亡后，或者还没下场时，它的控制器既不算“空”也不算“不在场”。结果：
     - `read_native_player_runtime` 把该方的整个 `abilityRuntime` 置为 null，overlay 显示“遥测不可用”，模型那一方也拿不到技能状态；
     - `resolve_ready_ability_request` 直接失败，报错 `could not resolve one ready native champion ability source`。
   - 牌组里有两个英雄，或者一个英雄加一个冠军时，只要其中一个不在场，另一个就放不出技能。
   - **修复**：把英雄形态的 ID 为 0 也视为合法的“不在场”控制器；读取状态时跳过它，不再让整方失效。
2. **按钮完全依赖严格校验的富遥测（rich telemetry）。**
   - `observe-rich` 校验 4000 多行的遥测，只要任何一个字段不合格（例如新版本英雄的某个特效），整个调用就会抛错，技能按钮随之被禁用。
   - **修复**：probe 的 `activate-ability` 新增 `OWNER -3 0` 形式，由 probe 自己选出该方最新一个可以释放的英雄。遥测异常，或者同时有多个可释放来源时，按钮仍然可用，会走这条后备路径。
3. **直接点游戏画面时没有技能操作。** 现在没选牌时点自己的英雄就能放技能（第 1.2 节）。在 MuMu 上，overlay 透明层会先截住鼠标点击，所以仍然要用侧边栏的技能按钮。

另外，控制台里写死的提示文字“勇者骑士技能已提交”改成了显示实际的技能名。

仍然会正常拦截的情况（属于正确行为，不是 bug）：圣水不够（`NotEnoughElixir`）、冷却中、次数用完、英雄还在部署动画中。

### 2.2 怎么判断是哪一种情况

在出问题的对局里运行：

```powershell
.\.venv\Scripts\python.exe -m native_runner.hero_diagnose > hero-state.json
```

输出内容包括：

- 每张手牌实际被发成了什么形态（`BasicForm`、`EvoForm` 或 `HeroForm`）；
- 双方技能控制器的原始状态；
- 富遥测校验是否通过，不通过时的原因。

### 2.3 英雄模型不显示：原因已确认，以及修复方法

**原因**：英雄形态的本体是 3D 模型。以骑士英雄为例，`knight_hero.toml` 里写的是：

```
PrefabAsset = "sc3d/characters/hero/hero_knight/hero_prefab_knight_battle"
```

它对应两类描述文件：一个 prefab 文件 `hero_prefab_*.json`，以及动画状态机文件 `asm_*.json`。这些文件在 `update/fingerprint.json` 里都有记录，但两边都拿不到：

- 原版 APK 只打包了模型和动画本体（`*.glb`），没有打包这些 `.json`；
- 原版游戏只在需要时才在线下载它们，第一次进大厅的那轮更新并不会下载；
- 离线引擎被断网，永远下载不到。

文件缺失时，3D 引擎组装不出这个模型，所以英雄本体看不见。英雄的逻辑、技能、投射物都是 2D 资源或纯逻辑，所以仍然正常。
已经实测确认：设备上的 `update/sc3d` 里只有少量特效、`hero_berserker` 和 `hero_valkyrie`，找不到任何 `hero_knight` 文件。

**修复**：用 `native_runner/update_assets.py` 把缺失的文件补进离线实例。它会：

1. 对照设备自己的 `fingerprint.json`，列出 `sc3d/` 下缺失或内容不一致的文件；
2. 从你指定的来源取回这些文件；
3. 逐个用 fingerprint 里的 SHA-1 校验，不一致的直接丢弃；
4. 推送到 `update/` 目录，并恢复游戏应用的文件属主和 SELinux 标签。

```powershell
# 1) 先看缺多少文件。它还会统计设备上已有文件的 SHA-1 是否和 fingerprint 一致，借此确认校验方式没问题
.\.venv\Scripts\python.exe -m native_runner.update_assets status

# 2) 选一个来源补齐（三选一）
.\.venv\Scripts\python.exe -m native_runner.update_assets sync --from-serial 127.0.0.1:<另一实例端口>
.\.venv\Scripts\python.exe -m native_runner.update_assets sync --from-dir D:\some\update-copy
.\.venv\Scripts\python.exe -m native_runner.update_assets sync --content-url https://<内容CDN根地址>
```

三种来源的说明：

- **`--from-serial`（推荐）**
  1. 另开一个 MuMu 实例，开启 root，装上**原版** Null's Royale，并保持联网。
  2. 进入大厅，在卡牌收藏里逐个打开英雄的 3D 预览；也可以用英雄联网打一局，让游戏把这些资源缓存下来。
  3. 对离线实例执行 sync，参数填这个实例的 adb 地址。
  4. 不要直接在离线实例上解除断网：离线 APK 跳过了在线加载流程，联网也不会去下载这些文件。
- **`--from-dir`**：已经有一份完整的 `update` 目录拷贝时使用，目录结构要和 `update/` 一致。
- **`--content-url`**：知道游戏的内容 CDN 地址时，会按 Supercell 惯用的 `<根地址>/<fingerprint sha>/<路径>` 直接下载。CDN 地址可以在原版游戏联网更新时，用 `adb logcat` 找下载请求的 URL。

`status` 会自动读取原版 APK（默认 `user-provided/nulls-royale.apk`，或 `.env` 里的 `CR_INPUT_APK`），游戏能从 APK 里直接读取的文件不算缺失；默认按目录汇总显示，加 `--all` 才逐个列出。补齐后重启离线游戏再开局。如果 `status` 显示的已有文件里有大量 SHA-1 不一致，先别 sync，把输出发给我。

## 3. 新增的控制命令（probe 控制端口）

| 命令 | 作用 |
| --- | --- |
| `touch status` | 查看触摸状态（新增字段：`ownerFilter`、`abilityTap`、`calibrating`、`customLayout`、`layout`、`lastDown`、`lastUp`、`abilityTaps*`） |
| `touch layout <18 个整数>` / `touch layout default` | 设置或恢复触摸布局，字段顺序见 `native_touch_layout.h` |
| `touch owner -1\|0\|1` | 限制屏幕触摸能控制的一方；-1 表示双方都可以 |
| `touch ability on\|off` | 开启或关闭“点英雄放技能” |
| `touch calibrate on\|off` | 校准模式：只记录点击坐标，不执行任何操作 |
| `activate-ability OWNER -3 0` | 释放该方最新一个可以释放的英雄或冠军的技能 |

## 4. 验证情况

- 新改的 probe 源码在主机上用 aarch64 交叉编译检查通过（`-Wall -Wextra` 无警告）。用的是 glibc 头文件加 Android 头文件桩，不是 NDK，所以正式构建仍然要用 `build_offline_engine.ps1`。
- 新增的测试：
  - `test_native_touch_phone.py`：编译并运行触摸布局的 C++ 测试，以及“点英雄选最近可释放来源”的 C++ 测试桩；
  - `test_phone_touch.py`：校准算法、布局校验、命令格式；
  - 原有的技能和 overlay 测试也做了相应更新。
- **没有在任何真机或 MuMu 上运行过**，需要你重新构建 APK 后实测。
