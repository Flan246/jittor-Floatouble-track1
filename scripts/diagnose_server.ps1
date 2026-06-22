# 服务器环境诊断脚本
# 运行方式：在 PowerShell 中执行：.\scripts\diagnose_server.ps1
# 该脚本通过你已有的 SSH 别名 aliyun-jittor 连接服务器并收集环境信息

$RemoteHost = "aliyun-jittor"
$OutputFile = "$env:TEMP\server_diagnose.log"

Write-Host "正在连接 $RemoteHost 收集环境信息..." -ForegroundColor Cyan

$DiagCommands = @"
#!/bin/bash
echo '===== OS Info ====='
lsb_release -a 2>/dev/null || cat /etc/os-release
echo ''
echo '===== GPU Info ====='
nvidia-smi
echo ''
echo '===== CUDA Version ====='
nvcc --version 2>/dev/null || echo 'nvcc not found'
echo ''
echo '===== Python Info ====='
which python3 && python3 --version
echo ''
echo '===== Conda Info ====='
which conda && conda --version || echo 'conda not found'
echo ''
echo '===== Disk Space ====='
df -h
echo ''
echo '===== Memory ====='
free -h
echo ''
echo '===== Existing Project Dir ====='
ls -la /root/jittor_competition 2>/dev/null || echo 'project dir not found'
"@

$DiagCommands | ssh $RemoteHost "bash -s" | Tee-Object -FilePath $OutputFile

Write-Host ""
Write-Host "诊断完成，日志已保存到: $OutputFile" -ForegroundColor Green
Write-Host "请把上面的输出完整复制给我。" -ForegroundColor Green
