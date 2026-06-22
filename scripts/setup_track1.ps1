# 赛道一阿里云 A10 一键部署脚本（本地运行）
# 运行方式：在 PowerShell 中执行 .\scripts\setup_track1.ps1

$ErrorActionPreference = "Stop"

$LocalProjectDir = "D:\cursor_file\计图"
$RemoteHost = "aliyun-jittor"
$RemoteProjectDir = "/root/jittor_competition"
$TempDir = "$env:TEMP\jittor_competition_upload"
$TarFile = "$env:TEMP\jittor_competition.tar.gz"
$RemoteScriptLocal = "$PSScriptRoot\setup_track1_remote.sh"

function Test-SSH {
    Write-Host "测试 SSH 连接 $RemoteHost ..." -ForegroundColor Cyan
    ssh $RemoteHost "echo 'SSH OK'"
    if ($LASTEXITCODE -ne 0) {
        throw "无法通过 SSH 连接 $RemoteHost，请检查 SSH 别名或密钥配置。"
    }
}

function Prepare-UploadPackage {
    Write-Host "准备上传包（排除 .git 和 __pycache__）..." -ForegroundColor Cyan
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir
    }
    New-Item -ItemType Directory -Path $TempDir | Out-Null

    # 使用 robocopy 镜像项目，排除 .git 和 __pycache__
    $robocopyArgs = @(
        '"' + $LocalProjectDir + '"',
        '"' + $TempDir + '"',
        '/MIR',
        '/XD', '.git', '__pycache__',
        '/XF', '*.pyc'
    )
    $robocopyCmd = "robocopy " + ($robocopyArgs -join ' ')
    Invoke-Expression $robocopyCmd | Out-Null

    if (Test-Path $TarFile) {
        Remove-Item -Force $TarFile
    }

    # 打包
    $parent = Split-Path $TempDir -Parent
    $folder = Split-Path $TempDir -Leaf
    tar -czf $TarFile -C $parent $folder

    $size = (Get-Item $TarFile).Length / 1MB
    Write-Host ("打包完成: {0:N1} MB" -f $size) -ForegroundColor Green
}

function Upload-Package {
    Write-Host "上传项目包到服务器..." -ForegroundColor Cyan
    scp $TarFile "${RemoteHost}:/tmp/jittor_competition.tar.gz"

    Write-Host "解压项目包到 $RemoteProjectDir ..." -ForegroundColor Cyan
    ssh $RemoteHost "rm -rf $RemoteProjectDir && mkdir -p $RemoteProjectDir && tar -xzf /tmp/jittor_competition.tar.gz -C $RemoteProjectDir --strip-components=1"
}

function Upload-RemoteScript {
    Write-Host "上传远程安装脚本..." -ForegroundColor Cyan
    scp $RemoteScriptLocal "${RemoteHost}:/tmp/setup_track1_remote.sh"
    ssh $RemoteHost "chmod +x /tmp/setup_track1_remote.sh"
}

function Run-RemoteSetup {
    Write-Host "开始执行远程环境配置（约 15–30 分钟，请不要关闭窗口）..." -ForegroundColor Cyan
    ssh -t $RemoteHost "bash /tmp/setup_track1_remote.sh"
}

# 主流程
try {
    Test-SSH
    Prepare-UploadPackage
    Upload-Package
    Upload-RemoteScript
    Run-RemoteSetup

    Write-Host ""
    Write-Host "===== 部署完成 =====" -ForegroundColor Green
    Write-Host "服务器项目目录: $RemoteProjectDir" -ForegroundColor Green
    Write-Host "你可以用以下命令再次登录: ssh $RemoteHost" -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "===== 部署出错 =====" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
