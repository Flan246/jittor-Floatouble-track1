# 在 D:\miniconda3 安装并验证 jittorgeometric 环境（路径须为纯 ASCII）
$ErrorActionPreference = "Stop"
$condaRoot = "D:\miniconda3"
$envName = "jittorgeometric"
$installer = "$env:USERPROFILE\Miniconda3-installer.exe"

if (-not (Test-Path "$condaRoot\Scripts\conda.exe")) {
    Write-Host "[1/4] Installing Miniconda to $condaRoot ..."
    if (-not (Test-Path $installer)) {
        Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $installer -UseBasicParsing
    }
    Start-Process -Wait -FilePath $installer -ArgumentList "/InstallationType=JustMe","/RegisterPython=0","/S","/D=$condaRoot"
}

$conda = "$condaRoot\Scripts\conda.exe"
$py = "$condaRoot\envs\$envName\python.exe"

Write-Host "[2/4] Accepting conda ToS (non-interactive) ..."
& $conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null
& $conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>$null
& $conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>$null

Write-Host "[3/4] Creating conda env $envName (python=3.10) ..."
& $conda create -n $envName python=3.10 -y

$env:JITTOR_HOME = "D:\jittor_cache"
New-Item -ItemType Directory -Force -Path "D:\jittor_cache" | Out-Null

Write-Host "[4/5] Installing jittor 1.3.7.16 ..."
& $py -m pip install -U pip
& $py -m pip install "numpy==1.26.4" "jittor==1.3.7.16"

Write-Host "[5/5] Verifying (first import ~2 min, downloads MSVC) ..."
& $py -c "import os; os.environ['JITTOR_HOME']=r'D:\jittor_cache'; import jittor as jt; print('jittor', jt.__version__); print('has_cuda', jt.has_cuda); print('sum', jt.array([1.,2.,3.]).sum().item())"
Write-Host "Done. Python: $py  |  set JITTOR_HOME=D:\jittor_cache"
