param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
. (Join-Path $PSScriptRoot "local_config.ps1")
$repository = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repository

$python = Get-LocalSetting "CR_PYTHON"
$checkpoint = "checkpoints/My/update-00000048.pt"

Write-Host "正在开启实战网络模式并转发控制端口..."
& $python -m native_runner.live_diagnose start

Write-Host "正在启动 AI 友谊战对战监听（权重：$checkpoint，门控温度：0.2）..."
& $python -u -m native_runner.live_play run --checkpoint $checkpoint --gate-temperature 0.2 @ExtraArgs
