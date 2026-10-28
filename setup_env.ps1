param([switch]$Cpu, [string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
}
$TaskPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# PyTorch与torchvision使用配套版本；CPU和CUDA环境通过同一开关选择下载源。
$TorchIndex = if ($Cpu) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu128" }
& $TaskPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url $TorchIndex --force-reinstall
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& $TaskPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $TaskPython -c "import torch,cv2; print('torch',torch.__version__,'OpenCV',cv2.__version__,'CUDA',torch.cuda.is_available())"
