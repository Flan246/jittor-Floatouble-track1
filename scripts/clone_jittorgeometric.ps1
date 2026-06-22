# 克隆 JittorGeometric（浅克隆，优先国内代理）
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "JittorGeometric"

if (Test-Path (Join-Path $dest ".git")) {
    Write-Host "已存在，执行 git pull ..."
    git -C $dest pull --ff-only
    exit 0
}

$urls = @(
    "https://ghproxy.net/https://github.com/AlgRUC/JittorGeometric.git",
    "https://github.com/AlgRUC/JittorGeometric.git"
)
foreach ($u in $urls) {
    Write-Host "Trying: $u"
    git clone --depth 1 $u $dest
    if ($LASTEXITCODE -eq 0) {
        git -C $dest log -1 --oneline
        Write-Host "Clone OK -> $dest"
        exit 0
    }
    Remove-Item -Recurse -Force $dest -ErrorAction SilentlyContinue
}
Write-Error "克隆失败，请检查网络或手动下载 ZIP"
