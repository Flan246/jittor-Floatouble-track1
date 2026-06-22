# 创建 track1/data → ../data_A 目录联接（仅需执行一次）
$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$link = Join-Path $repo "track1\data"
$target = Join-Path $repo "data_A"

if (Test-Path $link) {
    $item = Get-Item $link -Force
    if ($item.LinkType -eq "Junction") {
        Write-Host "已存在联接: $link -> $($item.Target)"
        exit 0
    }
    Write-Error "track1\data 已存在且不是联接，请手动处理后重试"
}

if (-not (Test-Path $target)) {
    Write-Error "目标不存在: $target"
}

New-Item -ItemType Junction -Path $link -Target $target | Out-Null
Write-Host "已创建: $link -> $target"
