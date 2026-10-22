# OpenCV去噪对目标检测的影响

计算机视觉第三章作业2。分支：hw2-opencv-denoising-detection。

固定COCO val2017中128张图像、YOLO11n权重、六个处理选项、六种噪声强度和三个噪声种子；另有六组干净图像对照。完整矩阵114组、14,592次图像推理。算法参数不按评测结果调整。

阅读 report.pdf 或 report.md；完整数值与原始预测见 results/full。个人PyCharm导读在本地两个仓库之外，不属于本分支。

## 环境与运行

已执行环境：Windows 11、Python 3.13.6、RTX 5070 Ti、PyTorch 2.7.1+cu128。建议Python 3.13，GPU使用匹配驱动；CPU可运行同一套算法，耗时与GPU记录不混合。

在本目录的PowerShell依次运行：

    powershell -ExecutionPolicy Bypass -File setup_env.ps1
    .\.venv\Scripts\python.exe prepare.py --config configs/full.json
    .\.venv\Scripts\python.exe verify.py
    .\.venv\Scripts\python.exe demo.py --stage detect --output practice-output/demo
    .\.venv\Scripts\python.exe run_experiment.py --limit 8 --condition-limit 8 --output practice-output/small
    .\.venv\Scripts\python.exe run_experiment.py --config configs/full.json --output practice-output/full
    .\.venv\Scripts\python.exe generate_report.py --input practice-output/full --output practice-output/report

setup_env.ps1 -Cpu 安装CPU PyTorch。固定GPU/CPU在配置中使用device值0或cpu；auto先做真实模型预热，能运行时选择GPU，出现可捕获的兼容错误时整体改用CPU。

正式结果已位于results/full。自己的运行使用practice-output，避免覆盖正式证据。分阶段学习使用demo.py的read、noise、denoise、detect。椒盐参数示例：--noise salt_pepper --level 0.03；level=0.03表示期望污染3%的像素位置，三个颜色通道共同置黑或白。

断点续跑按完整条件保存。相同输入再次运行会复用已完成条件；源代码、数据、模型、配置或设备变化会拒绝复用，需换输出目录。GPU浮点运算和不同软件环境可能带来微小数值差异，噪声数组的随机性则独立、确定。

generate_report.py使用本次固定课程设计的正式模板，校验128张、原始噪声/滤波/检测参数与模型哈希。修改参数的探索实验查看自己的CSV与图像；不能套用该模板中的固定参数说明。

## 文件与指标

| 文件 | 职责 |
|---|---|
| prepare.py、visionlab/data.py | 固定抽样、下载、缩放图像与真值、模型校验 |
| demo.py、visionlab/core.py | 分阶段体验、噪声和滤波、质量计算、固定检测器 |
| run_experiment.py、visionlab/experiment.py | 成对实验、计时、保存记录、恢复、汇总 |
| visionlab/evaluation.py | 官方COCO AP与固定阈值TP/FP/FN |
| generate_report.py | 从已保存记录重建图表、案例与中文报告 |
| tools/md_to_pdf.py | Markdown转PDF，Windows默认使用宋体；其他系统指定--font |
| verify.py | 有意义的配对噪声、坐标、类别及crowd评价检查 |

AP采用推理置信度0.001以上的预测、COCO官方101点插值和最大100个检测；P/R/F1使用置信度0.25、同类别IoU0.5匹配并忽略crowd。NMS阈值0.7与匹配IoU是不同操作。PSNR/SSIM对照为缩放后干净图像。clean+none的PSNR是正无穷，在JSON存null，在报告显示∞。

## 来源与发布范围

图像和标注来自COCO官方。results/full/data_manifest.json列出128个ID、来源、许可、文件哈希与缩放比例；图像各有原作者许可，展示案例保留COCO来源说明。模型来自Ultralytics官方v8.3.0发布资产，下载后核验SHA256。

依赖Ultralytics及其预训练模型遵循其AGPL-3.0许可。课程实现的源码按AGPL-3.0发布，详见LICENSE；论文及COCO图像的版权归各自作者。

不纳入Git：虚拟环境、下载图像、模型、缓存、IDE配置、练习输出。保留正式配置、标注与清单、压缩逐图预测、指标、图示和报告。两个作业不自动合并到main。
