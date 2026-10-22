# 作业1：CVPR论文中的图像处理方法与作用

本分支：hw1-cvpr-paper-analysis。

主分析DCP-SFR（CVPR 2026）与UniRestore（CVPR 2025），结合第三章课件和当前工业异常检测构想，讨论任务信息保留。

- report.pdf：正式中文报告。
- report.md：可编辑报告。
- figures/：原论文方法图节选，来源标注在正文。
- references.bib和source_manifest.json：文献与全文校验信息。

文献实验数字为作者报告，本作业未训练或复现两篇完整模型。具体研究方案标注为待验证假设。全文缓存位于被Git忽略的.cache/。

已阅读DCP-SFR出版版10页、UniRestore出版版11页及补充材料7页。补充材料中不同任务提示的PSNR/分类对照与实际计算开销已纳入分析。图示版权归原论文作者，正文逐图标明出处。

重新生成PDF：安装requirements.txt后，在本目录运行python tools/md_to_pdf.py report.md report.pdf。Windows默认嵌入宋体；其他系统用--font指定支持中文的TrueType字体。

作业2位于hw2-opencv-denoising-detection分支。
