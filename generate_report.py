"""Generate all figures and report numbers from saved predictions, without inference."""
import argparse
from collections import Counter
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from visionlab.core import DEFAULT_FILTERS, METHOD_NAMES, add_noise, denoise, draw_boxes, read_image
from visionlab.data import WEIGHT_SHA256, save_json
from tools.md_to_pdf import render
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
METHODS = ["none", "mean", "gaussian", "median", "bilateral", "nlm"]
COLORS = ["#4B5563", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def metric(row, name, digits=3, percent=False):
    m, s = row[name+"_mean"], row[name+"_std"]
    if m is None:
        return "∞"
    # 检测比率展示为百分数，原始记录仍保留0到1的数值。
    scale = 100 if percent else 1
    return f"{m*scale:.{digits}f}±{s*scale:.{digits}f}"

def table(rows, headers):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join(["---"]*len(headers))+" |"] +
                     ["| "+" | ".join(map(str, r))+" |" for r in rows])

def matched_ids(truth, row, confidence=.25):
    """Use official matching to explain which ordinary GT objects were lost/gained."""
    if not row["predictions"]:
        assert row["tp"]==0
        return set()
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=truth; gt.createIndex()
        dt=gt.loadRes(row["predictions"])
        ev=COCOeval(gt,dt,"bbox")
        ev.params.imgIds=[row["image_id"]]
        ev.evaluate()
    # 案例分析复用官方匹配规则，避免画框展示与统计结果口径不同。
    matched=set()
    for item in ev.evalImgs:
        if item is None or item["aRng"]!=ev.params.areaRng[0]:
            continue
        mask=(np.asarray(item["dtScores"])>=confidence)&~item["dtIgnore"][0]
        matched.update(int(i) for i in item["dtMatches"][0][mask] if i>0)
    assert len(matched)==row["tp"]
    return matched

def make_plots(aggregate, figdir):
    # 按噪声、强度与方法索引汇总表，绘图时保持方法和颜色顺序一致。
    by = {(r["noise"], r["level"], r["method"]): r for r in aggregate}
    for name, levels, label in [("gaussian",[10,25,50],"高斯噪声标准差（0—255尺度）"),
                                ("salt_pepper",[.01,.03,.05],"椒盐污染比例（%）")]:
        fig, axes = plt.subplots(1, 2, figsize=(11.3,4.2))
        x = np.array(levels)*(100 if name=="salt_pepper" else 1)
        for method, color in zip(METHODS,COLORS):
            rows = [by[(name,l,method)] for l in levels]
            for ax, key, scale in [(axes[0],"map50_95",100),(axes[1],"ssim",1)]:
                ax.errorbar(x,[r[key+"_mean"]*scale for r in rows],
                            yerr=[r[key+"_std"]*scale for r in rows],
                            label=METHOD_NAMES[method], color=color, marker="o", capsize=3, linewidth=1.7)
                ax.set_xlabel(label)
                ax.grid(axis="y",alpha=.22)
        axes[0].axhline(by[("clean",0,"none")]["map50_95_mean"]*100,
                       color="#888888",linestyle="--",linewidth=1)
        axes[0].set_ylabel("mAP@0.5:0.95（%）")
        axes[1].set_ylabel("SSIM")
        axes[0].legend(ncol=3,fontsize=11.2,loc="lower left")
        fig.tight_layout()
        fig.savefig(figdir/(name+"_curves.png"),dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.2,4.7))
    for method,color in zip(METHODS,COLORS):
        rows = [r for r in aggregate if r["noise"]!="clean" and r["method"]==method]
        ax.scatter([r["psnr_db_mean"] for r in rows],[r["map50_95_mean"]*100 for r in rows],
                   label=METHOD_NAMES[method],color=color,s=65,alpha=.85)
    ax.set(xlabel="PSNR（dB；逐图平均后再跨种子平均）",ylabel="mAP@0.5:0.95（%）")
    ax.grid(alpha=.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(figdir/"quality_detection.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11.3,4.0))
    clean = [by[("clean",0,m)] for m in METHODS]
    x = np.arange(6)
    axes[0].bar(x,[r["map50_95_mean"]*100 for r in clean],color=COLORS)
    axes[0].set_ylabel("干净输入 mAP@0.5:0.95（%）")
    axes[0].set_ylim(0, max(r["map50_95_mean"]*100 for r in clean)*1.15)
    times = [[r["denoise_ms_mean"] for r in aggregate if r["method"]==m] for m in METHODS]
    axes[1].bar(x,[np.mean(t) for t in times],color=COLORS)
    axes[1].set_ylabel("去噪平均耗时（ms/张）")
    # NLM耗时远高于其他方法；symlog保留零附近的线性区间，便于同时查看各方法。
    axes[1].set_yscale("symlog",linthresh=1)
    for ax in axes:
        ax.set_xticks(x,[METHOD_NAMES[m] for m in METHODS],rotation=20)
        ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir/"clean_and_time.png",dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.7,3.3))
    ax.set(xlim=(0,10),ylim=(0,3))
    ax.axis("off")
    boxes = [(0.1,1.6,"COCO固定抽样\n128张＋真值"),
             (2.7,1.6,"最长边≤640\n精确缩放标注"),
             (5.3,1.6,"成对噪声\n6种强度×3种子"),
             (7.9,1.6,"六种处理\n含不去噪"),
             (7.9,.15,"PSNR / SSIM\nCPU质量评价"),
             (5.3,.15,"固定YOLO11n\n同一推理配置"),
             (2.7,.15,"COCO AP\nP/R/F1匹配"),
             (.1,.15,"条件记录与汇总\n重建图表／报告")]
    for x,y,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),1.95,1.0,boxstyle="round,pad=.08",
                                   facecolor="#EAF1F7",edgecolor="#53789A"))
        ax.text(x+.975,y+.5,label,ha="center",va="center",fontsize=11)
    for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]:
        xa,ya,_=boxes[a]; xb,yb,_=boxes[b]
        if ya==yb:
            start=(xa+1.95,ya+.5) if xb>xa else (xa,ya+.5)
            end=(xb,yb+.5) if xb>xa else (xb+1.95,yb+.5)
        else:
            start=(xa+.975,ya); end=(xb+.975,yb+1)
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color="#53789A",lw=1.6))
    fig.tight_layout()
    fig.savefig(figdir/"pipeline.png",dpi=180)
    plt.close(fig)

def case_figures(inp, config, truth, output):
    ids = [im["id"] for im in truth["images"]]
    image_by = {im["id"]:im for im in truth["images"]}
    def records(key):
        with gzip.open(inp/"records"/(key+".json.gz"),"rt",encoding="utf-8") as f:
            return {r["image_id"]:r for r in json.load(f)}
    cases = []
    # Selection is explicit and reproducible; examples illustrate mechanisms, not prevalence.
    for noise,level,method,direction in [("salt_pepper",.03,"median","benefit"),
                                         ("gaussian",25,"mean","cost")]:
        baseline = records(f"{noise}-{level}-s42-none")
        filtered = records(f"{noise}-{level}-s42-{method}")
        def change(i):
            b,f=baseline[i],filtered[i]
            return (f["tp"]-b["tp"])-(f["fp"]-b["fp"])
        ranked = sorted(ids,key=lambda i: (change(i),i))
        image_id = ranked[-1] if direction=="benefit" else ranked[0]
        clean = read_image(Path(config["data_dir"])/"clean"/image_by[image_id]["file_name"])
        noisy = add_noise(clean,noise,level,42,image_id)
        clean_records = records("clean-0-s0-none")
        anns = [a for a in truth["annotations"] if a["image_id"]==image_id]
        arrays = [clean,noisy] + [denoise(noisy,m,config["filters"]) for m in METHODS[1:]]
        titles = ["原图","噪声／不去噪"]+[METHOD_NAMES[m] for m in METHODS[1:]]
        fig,axes=plt.subplots(2,4,figsize=(12,7))
        for ax,arr,title in zip(axes.flat,arrays,titles):
            ax.imshow(arr[:,:,::-1]); ax.set_title(title); ax.axis("off")
        axes.flat[-1].axis("off")
        fig.suptitle(f"COCO ID {image_id} · {noise}={level} · seed42",fontsize=14)
        fig.tight_layout()
        fig.savefig(output/f"case_{direction}_pixels.png",dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4.4))
        for ax,arr,row,title in zip(axes,[clean,noisy,denoise(noisy,method,config["filters"])],
                                   [clean_records[image_id],baseline[image_id],filtered[image_id]],
                                   ["原图","不去噪",METHOD_NAMES[method]]):
            ax.imshow(draw_boxes(arr,anns,row["predictions"],.25)[:,:,::-1])
            ax.set_title(f'{title}\nTP={row["tp"]}, FP={row["fp"]}, FN={row["fn"]}')
            ax.axis("off")
        fig.suptitle("绿色：真值框；红色：预测框（类别ID:置信度）",fontsize=12)
        fig.tight_layout()
        fig.savefig(output/f"case_{direction}_boxes.png",dpi=180)
        plt.close(fig)
        targets=[a for a in anns if not a.get("iscrowd",0)]
        h,w=clean.shape[:2]
        if targets:
            a=min(targets,key=lambda a:a["bbox"][2]*a["bbox"][3])
            x,y,bw,bh=a["bbox"]
            pad=max(18,min(50,max(bw,bh)*.4))
            x0,y0=max(0,int(x-pad)),max(0,int(y-pad))
            x1,y1=min(w,int(x+bw+pad)),min(h,int(y+bh+pad))
        else:
            x0,y0,x1,y1=w//4,h//4,w*3//4,h*3//4
        fig,axes=plt.subplots(1,4,figsize=(11,3.5))
        for ax,arr,title in zip(axes,[clean,noisy,denoise(noisy,method,config["filters"]),
                                     denoise(noisy,"nlm",config["filters"])],
                               ["原图局部","噪声局部",METHOD_NAMES[method]+"局部","非局部均值局部"]):
            ax.imshow(arr[y0:y1,x0:x1,::-1],interpolation="nearest")
            ax.set_title(title); ax.axis("off")
        fig.tight_layout()
        fig.savefig(output/f"case_{direction}_crop.png",dpi=180)
        plt.close(fig)
        cases.append({"noise":noise,"level":level,"seed":42,"method":method,"image_id":image_id,
                      "selection":"extreme change in (TP-FP), tie by image ID",
                      "direction":direction,"change_tp_minus_fp":change(image_id),
                      "baseline":{k:baseline[image_id][k] for k in ["tp","fp","fn","psnr_db","ssim"]},
                      "filtered":{k:filtered[image_id][k] for k in ["tp","fp","fn","psnr_db","ssim"]},
                      "crop_xyxy":[x0,y0,x1,y1]})
        bm=matched_ids(truth,baseline[image_id])
        fm=matched_ids(truth,filtered[image_id])
        amap={a["id"]:a for a in anns}
        cnames={c["id"]:c["name"] for c in truth["categories"]}
        cases[-1].update({"lost_matched_gt_ids":sorted(bm-fm),"gained_matched_gt_ids":sorted(fm-bm),
                          "lost_classes":dict(Counter(cnames[amap[i]["category_id"]] for i in bm-fm)),
                          "gained_classes":dict(Counter(cnames[amap[i]["category_id"]] for i in fm-bm))})
    return cases

def main(inp, output, pdf=True):
    inp, output=Path(inp),Path(output)
    complete=load(inp/"completion.json")
    if not complete["full_matrix"]:
        raise ValueError("This report compares the complete 114-condition matrix; finish it first")
    aggregate=load(inp/"aggregate_results.json")
    env=load(inp/"environment.json")
    config=load(inp/"config.json")
    truth=load(inp/"annotations_resized.json")
    expected={"sample_count":128,"sample_seed":20260930,"max_side":640,"image_size":640,
              "inference_confidence":.001,"report_confidence":.25,"nms_iou":.7,"max_detections":100,
              "seeds":[42,43,44],"gaussian_sigmas":[10,25,50],
              "salt_pepper_probabilities":[.01,.03,.05],"methods":METHODS,"filters":DEFAULT_FILTERS}
    changed=[k for k,v in expected.items() if config.get(k)!=v]
    if changed or len(truth["images"])!=128 or env["model_sha256"]!=WEIGHT_SHA256:
        raise ValueError("The formal report template requires the fixed coursework design. "
                         "For modified exercises read their CSVs; changed settings: "+", ".join(changed))
    figdir=output/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    make_plots(aggregate,figdir)
    cases=case_figures(inp,config,truth,figdir)
    by={(r["noise"],r["level"],r["method"]):r for r in aggregate}
    conditions=[load(p) for p in sorted((inp/"conditions").glob("*.json"))]
    paired=[]
    for noise,levels in [("gaussian",[10,25,50]),("salt_pepper",[.01,.03,.05])]:
        for level in levels:
            for method in METHODS[1:]:
                diffs=[]
                for seed in [42,43,44]:
                    filt=next(r for r in conditions if (r["noise"],r["level"],r["seed"],r["method"])==(noise,level,seed,method))
                    none=next(r for r in conditions if (r["noise"],r["level"],r["seed"],r["method"])==(noise,level,seed,"none"))
                    diffs.append((filt["map50_95"]-none["map50_95"])*100)
                paired.append({"noise":noise,"level":level,"method":method,
                               "delta_ap_points_mean":float(np.mean(diffs)),
                               "delta_ap_points_std":float(np.std(diffs,ddof=1)),
                               "three_paired_deltas":diffs})
    counts={noise:sum(r["noise"]==noise and r["delta_ap_points_mean"]>0 for r in paired)
            for noise in ["gaussian","salt_pepper"]}
    moderate=[]
    for noise,level,label in [("gaussian",25,"高斯σ25"),("salt_pepper",.03,"椒盐3%")]:
        for method in METHODS:
            r=by[(noise,level,method)]
            moderate.append([label,METHOD_NAMES[method],metric(r,"map50",2,True),
                             metric(r,"map50_95",2,True),metric(r,"f1",2,True),
                             metric(r,"psnr_db",2),metric(r,"ssim",3)])
    clean_rows=[]
    for method in METHODS:
        r=by[("clean",0,method)]
        clean_rows.append([METHOD_NAMES[method],f'{100*r["map50_95_mean"]:.2f}',
                           f'{100*r["f1_mean"]:.2f}',"∞" if r["psnr_db_mean"] is None else f'{r["psnr_db_mean"]:.2f}',
                           f'{r["ssim_mean"]:.3f}'])
    threshold_rows=[]
    for noise,level,label in [("gaussian",25,"高斯σ25"),("salt_pepper",.03,"椒盐3%")]:
        for method in METHODS:
            r=by[(noise,level,method)]
            threshold_rows.append([label,METHOD_NAMES[method],metric(r,"precision",2,True),
                                   metric(r,"recall",2,True),metric(r,"f1",2,True)])
    best_rows=[]
    for noise,level in [("gaussian",10),("gaussian",25),("gaussian",50),
                       ("salt_pepper",.01),("salt_pepper",.03),("salt_pepper",.05)]:
        rows=[by[(noise,level,m)] for m in METHODS]
        best=max(rows,key=lambda r:r["map50_95_mean"])
        qbest=max(rows,key=lambda r:r["psnr_db_mean"])
        base=by[(noise,level,"none")]["map50_95_mean"]
        best_rows.append([f'{noise} {level}',METHOD_NAMES[best["method"]],
                          f'{best["map50_95_mean"]*100:.2f}',
                          f'{(best["map50_95_mean"]-base)*100:+.2f}',METHOD_NAMES[qbest["method"]]])
    timings=[]
    for method in METHODS:
        # 每种方法有1个干净组和6个噪声组；噪声组已先汇总三个种子，这里再按组同权平均。
        rows=[r for r in aggregate if r["method"]==method]
        dm=float(np.mean([r["denoise_ms_mean"] for r in rows]))
        im=float(np.mean([r["detection_ms_mean"] for r in rows]))
        timings.append([METHOD_NAMES[method],f'{dm:.2f}',f'{im:.2f}',f'{dm+im:.2f}'])
    total=len(truth["annotations"])
    nc=sum(not a.get("iscrowd",0) for a in truth["annotations"])
    categories=next(r["categories_with_ground_truth"] for r in conditions)
    clean_ap=by[("clean",0,"none")]["map50_95_mean"]*100
    version_table=table([["Python",env["python"]],["PyTorch",env["torch"]],["CUDA",env["cuda"]]],
                        ["环境项目","实际版本"])
    version_table+="\n\n"+table([[k,v] for k,v in env["versions"].items()],["软件","实际版本"])
    text=f"""# OpenCV去噪对目标检测的影响
## 计算机视觉第三章 · 作业2实验报告

实验共设置114组条件，完成14,592次图像推理，使用固定预训练模型。检测指标由逐图预测记录通过COCO官方评价程序计算。实验指纹：{complete["fingerprint"]}。

## 摘要

本实验比较均值、高斯、中值、双边和彩色非局部均值滤波与不去噪，在128张COCO验证图像上检验画质与检测精度的关系。噪声包含三档高斯噪声和三档椒盐噪声，每档使用三个种子，另加入六种干净输入对照。干净不滤波的mAP@0.5:0.95为{clean_ap:.2f}%。在每类噪声的15个“方法×强度”比较中，高斯噪声有{counts["gaussian"]}个、椒盐噪声有{counts["salt_pepper"]}个滤波选项的平均AP高于对应不去噪基线。收益取决于噪声模型、强度和滤波方式，图像质量指标不能代替任务评价。

## 1. 实验目的与问题

课程要求用OpenCV实现去噪，并简略比较不同方法对后续目标检测的影响。本实验进一步回答三个问题：去噪能否恢复被噪声破坏的目标证据；去噪是否同时抹掉小目标、轮廓与纹理；PSNR/SSIM提高是否对应检测改善。作业1讨论的UniRestore按任务恢复图像，DCP-SFR保留弱异常线索；本作业以传统方法建立可解释的像素处理对照。

工作假设是：中值对脉冲噪声通常更适合，平均操作可降低独立随机噪声，但可能模糊边界。这些是方法机理提出的预期，不作为实验排名。所有滤波参数在完整实验前固定，没有使用评测结果调参。

## 2. 噪声模型与去噪原理

### 2.1 高斯噪声和椒盐噪声

高斯噪声在各像素各颜色通道独立生成，标准差使用0—255尺度。先转浮点相加，再裁剪与四舍五入回uint8，避免8位整数回绕：

$$Y=\\mathrm{{round}}\\left(\\mathrm{{clip}}(X+\\epsilon,0,255)\\right),\\quad \\epsilon\\sim\\mathcal{{N}}(0,\\sigma_n^2)$$

σ_n取10、25、50。靠近0或255处裁剪会改变实际扰动分布，这里的σ_n描述裁剪前的噪声；裁剪后，靠近像素范围边界的扰动分布会发生变化。

椒盐噪声按像素位置抽一个u∈[0,1)，u<p/2时三个通道同时置0，p/2≤u<p时置255，其余保留原像素。p取0.01、0.03、0.05，是期望污染位置比例，有限图像中的实际比例会波动。两个模型均是合成退化，不代表真实相机全部噪声。

同一图像、噪声类型、强度与种子生成同一随机流。SHA256散列的前8字节生成NumPy随机数生成器种子；方法名称不参与散列，因而六个方法处理完全相同的噪声实例。

### 2.2 均值与高斯滤波

均值对邻域等权平均，是线性低通滤波。窗口越大通常越平滑，也越容易混合目标与背景：

$$\\hat{{X}}(p)=\\frac{{1}}{{|\\Omega|}}\\sum_{{q\\in\\Omega(p)}}Y(q)$$

高斯滤波仍是线性加权平均，但离中心越近权重越大。σ_f表示滤波核标准差，与生成噪声用的σ_n不同：

$$G(q)\\propto\\exp\\left(-\\frac{{\\|q\\|^2}}{{2\\sigma_f^2}}\\right),\\quad \\hat{{X}}=G*Y$$

二者适合降低局部随机波动，但不能判断高频是噪声还是细小目标。课程第30页将均值列入非线性，与第31/36页的线性滤波示例不一致；固定权重的均值与高斯在数学上均为线性操作。

### 2.3 中值滤波

中值用邻域排序后的中间值替代中心值，本实验按各通道处理：

$$\\hat{{X}}(p)=\\mathrm{{median}}\\{{Y(q):q\\in\\Omega(p)\\}}$$

少量极端黑白脉冲不容易改变排序中间值，因此中值适合椒盐噪声。它是非线性操作；细线、小点、狭窄目标也可能因邻域多数表决而消失。

### 2.4 双边滤波

双边同时考虑空间距离和颜色差异，通过归一化权重平均：

$$\\hat{{X}}(p)=\\frac{{\\sum_q w_s(p,q)w_c(p,q)Y(q)}}{{\\sum_q w_s(p,q)w_c(p,q)}}$$

空间权重与颜色权重均按高斯形式衰减，后者由当前图像内容决定，因此为非线性滤波。颜色差大的真实边缘可以少混合，但严重噪声造成的颜色差也可能被当成边缘保留下来。

### 2.5 彩色非局部均值

非局部均值比较图像块而非只比较单个像素，在搜索区域寻找外观类似的块：

$$\\hat{{X}}(p)=\\sum_q w(p,q)Y(q),\\quad w(p,q)\\propto\\exp\\left(-\\frac{{d(P_p,P_q)}}{{h^2}}\\right)$$

这是解释机理的简化表达；OpenCV彩色实现内部转到Lab并分别处理亮度与颜色，采用快速算法。本实验调用fastNlMeansDenoisingColored完成处理，以上公式仅用于说明其加权思想。相似纹理有利于平均抑噪；h过强会损失细节，h固定而噪声很强时又可能去除不足。搜索窗口较大使其耗时明显增加。

| 方法 | 本实验固定参数 | 主要取舍 |
|---|---|---|
| 不去噪 | 复制输入 | 保留噪声与全部原细节 |
| 均值 | 3×3 | 低成本平滑，边界可能模糊 |
| 高斯 | 3×3，σ_f=1 | 中心权重较大，仍有细节损失 |
| 中值 | 3×3 | 对脉冲稳健，窄结构可能消失 |
| 双边 | d=5，σColor=25，σSpace=25 | 边缘保护依赖当前颜色差 |
| 彩色非局部均值 | h=10，hColor=10；模板7、搜索21 | 利用块相似性，计算开销较大 |

均值、高斯、双边使用BORDER_REFLECT_101；中值与非局部均值沿用OpenCV内部边界约定。全部处理彩色BGR uint8数组，输出形状保持一致。

## 3. 数据、环境与实验流程

### 3.1 数据来源与固定抽样

从官方COCO val2017的5,000张图像按图像ID排序，用NumPy default_rng(20260930)无放回抽取128个索引，再按ID排序保存。共有{total}个标注，其中{nc}个非crowd对象；具有非crowd真值的类别为{categories}/80。未出现真值的类别按COCO默认方式不计入AP均值，因此不能把本子集AP与整个COCO榜单直接比较。

最长边缩至最多640，不放大较小图像，INTER_AREA插值。由于尺寸取整，分别记录sx=新宽/原宽、sy=新高/原高，真值[x,y,w,h]分别乘sx、sy；area乘sx×sy，保留iscrowd。评价是bbox，删除不参与本作业的分割多边形。PNG避免二次JPEG压缩。

data_manifest.json保留128个ID、下载URL、原图/处理图SHA256、许可与精确缩放比例；annotations_resized.json保存正式真值。COCO图像版权与使用许可归原作者，案例图按清单中的COCO来源追溯。所有噪声实验共用该子集，未训练检测器或用验证集选参数。

### 3.2 实际执行环境

硬件：{env["gpu"]}；全部正式条件使用设备{env["device"]}，无CPU/GPU结果混用。系统：{env["platform"]}。推理精度float32，OpenCV固定单线程，PyTorch CPU线程数4。

{version_table}

初建本地虚拟环境时为复用已安装依赖启用了system-site-packages，后在该环境中单独安装CUDA 12.8版PyTorch。交付的setup_env.ps1为新机器创建普通隔离虚拟环境，不要求使用原系统依赖。requirements.txt固定直接依赖，环境记录保存实际版本，首次联网安装过程受镜像与驱动可用性影响。

模型为Ultralytics官方YOLO11n COCO预训练权重，来源v8.3.0发布资产。SHA256：{env["model_sha256"]}。固定imgsz640、rect=False、augment=False、NMS IoU0.7、max_det100、推理置信度0.001。模型API返回映射回处理图像的xyxy，转换为COCO xywh；YOLO的0—79类别索引映射为完整COCO类别ID列表，不能简单加1。

### 3.3 流程与分组

![实验流程](figures/pipeline.png)

图1：正式执行流程。同一图像、强度与种子下，各方法重建相同的噪声输入并独立处理；检测器参数全程固定。

6组干净输入（原图与五种滤波）＋2类噪声×3强度×3种子×6方法=114个条件，每条件128张图像。先完成3项基础验证、单图六方法检测及8张×8条件小规模实验，随后运行完整矩阵。

入口依次为prepare.py准备数据、demo.py分阶段演示、run_experiment.py批量运行、generate_report.py汇总已有预测。报告生成不重新推理，而是从records中的预测框重建检测展示。每条件原子保存压缩逐图记录与JSON摘要。配置、数据、模型、设备和核心代码哈希共同构成指纹；不匹配时拒绝恢复，防止不同实验混合。

### 3.4 指标定义与计时边界

PSNR=10log10(255²/MSE)，MSE对全部像素和三个颜色通道计算，先转float64防止uint8减法回绕。SSIM用scikit-image、data_range255、channel_axis2，默认7×7窗口及其他默认参数；各图同权平均。干净不处理的MSE=0、PSNR=∞，JSON使用null表示此特殊值，不能把它解释为缺测或0 dB。

mAP@0.5与mAP@0.5:0.95使用官方pycocotools.COCOeval bbox、101点召回插值、同类别匹配、crowd忽略与maxDets100。后者平均IoU0.50至0.95、步长0.05；AP使用置信度0.001以上的全部保留预测，并非只使用0.25以上预测。

固定阈值Precision=TP/(TP+FP)、Recall=TP/(TP+FN)、F1=2PR/(P+R)，在置信度≥0.25、IoU≥0.5下按COCO逐类别匹配，合并所有图像计算micro指标。重复预测只允许一个匹配同一普通真值，错类别不能匹配；crowd忽略项不作为FP/FN。先每种子计算整个子集指标，再对三个种子取均值与样本标准差（ddof=1），不是对逐图AP平均。

去噪时间仅覆盖滤波调用。检测计时从模型调用前开始，到预测框、分数和类别转为CPU上的NumPy数组后结束，GPU在计时前后同步；随后将xyxy转为xywh并组装预测字典的Python循环不在该区间内。磁盘读写、质量评价、绘图和三次预热均不计入耗时。记录逐图平均与中位数。正式方法按固定顺序运行；期间同时进行文档生成和少量演示验证，背景负载、温度、调度均可能影响耗时，所以其为本机工程记录，未作为严格微基准或显著性检验。误差条为三个噪声种子的标准差，不能解释为跨数据集泛化置信区间。

## 4. 实验结果与可视化

### 4.1 干净输入对照

{table(clean_rows,["处理","AP50:95（%）","F1（%）","PSNR（dB）","SSIM"])}

表1：干净对照只有一组确定输入，不绘制跨噪声种子的误差条。直接滤波会改变边界与纹理，必须保留该对照，才能判断算法自身带来的代价。

![干净对照和时间](figures/clean_and_time.png)

图2：左为干净对照AP，右为每种方法在7个汇总组上的平均去噪耗时，包括1个干净组及6个噪声类型与强度组合；每个噪声组先对三个种子取均值。右图使用对称对数坐标，线性区间0—1 ms；柱高不能按线性比例直接比较。

### 4.2 两类噪声和强度趋势

![高斯噪声结果](figures/gaussian_curves.png)

图3：高斯噪声下AP与SSIM，点为三个种子均值，误差条为标准差。AP面板虚线为干净不处理参考。

![椒盐噪声结果](figures/salt_pepper_curves.png)

图4：椒盐噪声下AP与SSIM，横轴以百分比标注。三个种子改变噪声的随机分布位置，图像子集与检测器保持不变。

{table(best_rows,["输入","最高平均AP选项","AP50:95（%）","较不去噪Δ百分点","最高PSNR选项"])}

表2：按预先固定的选项做事后描述，不据此回调参数。没有额外调参集；“最高”只针对本子集、固定参数与所列方法。不同噪声强度的首选可能变化。

### 4.3 中等强度的详细对比

{table(moderate,["输入","方法","AP50（%）","AP50:95（%）","F1（%）","PSNR（dB）","SSIM"])}

表3：均值±样本标准差。AP/F1以百分数表达，差值讨论使用百分点。完整42行汇总（6行干净＋36行噪声方法组）见aggregate_results.csv，114行独立条件见condition_results.csv。表中显示为0.00或0.000的标准差经过四舍五入，完整数值以CSV为准。

{table(threshold_rows,["输入","方法","Precision（%）","Recall（%）","F1（%）"])}

表4：固定置信度0.25、IoU0.5下的micro指标；AP与固定阈值F1可给出不同排序。噪声造成置信度或定位改变，均可能改变TP/FP/FN。

### 4.4 图像质量与检测关系

![质量与检测散点](figures/quality_detection.png)

图5：每点为一个噪声类型、强度、方法的三种子均值，共36点。不同点的输入强度并不相同；该图用于显示二者非一一对应，不能把跨条件相关性解释为画质变化的因果系数。

PSNR惩罚每个像素误差，检测器更关心可分辨的类别与空间结构。抹平噪声也可能抹掉小目标；少量高频残差可能降低PSNR但保留检测证据。双边与非局部均值的固定强度参数不随噪声调整，不能把不适配某一档噪声归为方法普遍无效。

### 4.5 案例分析：收益与代价

案例选择规则在生成程序中固定：seed42、椒盐3%中值选择相对不去噪(TP−FP)变化最大的一张；高斯σ25均值选择变化最小的一张；若变化值相同，收益案例取较大图像ID，代价案例取较小图像ID。它们是极端示例，用于展示误检漏检机制，不能代表发生比例，也不据此宣布某方法总体优越。

"""
    # 将类别计数转为文字，避免报告正文直接展示Python字典。
    def class_counts(counts):
        return "、".join(f"{name} {count}个" for name,count in counts.items()) or "无"

    for c in cases:
        d=c["direction"]; b,f=c["baseline"],c["filtered"]
        explanation=(f'本例新增{len(c["gained_matched_gt_ids"])}个真值匹配，同时失去{len(c["lost_matched_gt_ids"])}个原有匹配；总体收益不能解释为每个目标都获益。'
                     if d=="benefit" else
                     f'本例失去{len(c["lost_matched_gt_ids"])}个原有匹配，类别为{class_counts(c["lost_classes"])}。局部图中很小的轮廓被均值平滑，提示小目标证据与随机高频同时被处理；具体定位或分数变化仍应按保存的框与置信度核查。')
        text+=f"""### 案例：COCO {c["image_id"]}，{c["noise"]}={c["level"]}，{METHOD_NAMES[c["method"]]}

不去噪TP/FP/FN={b["tp"]}/{b["fp"]}/{b["fn"]}，滤波后={f["tp"]}/{f["fp"]}/{f["fn"]}；(TP−FP)变化{c["change_tp_minus_fp"]:+d}。局部裁剪坐标为{c["crop_xyxy"]}，以最小面积非crowd真值框外扩选取。所有预测框来自正式记录，未为图示重新推理。

按官方匹配追踪同一真值，新增匹配的目标为{class_counts(c["gained_classes"])}；失去原有匹配的目标为{class_counts(c["lost_classes"])}；具体真值ID保存在analysis_summary.json。这里同时涉及定位、类别和置信度变化，不能只看清晰度或仅凭框图确定原因。

{explanation}

![像素对照](figures/case_{d}_pixels.png)

图：原图、噪声与五种处理。图像为COCO val2017，对应来源URL与许可见data_manifest.json该ID。

![预测框对照](figures/case_{d}_boxes.png)

图：绿色为非crowd真值，红色为置信度≥0.25预测；重叠框不等于成功匹配，还须类别相同与IoU达标。红字为COCO类别ID和置信度。

![局部放大](figures/case_{d}_crop.png)

图：同一坐标区域按最近邻展示，观察轮廓、细纹与残留脉冲。局部锐利度只能作为视觉线索，实际检测判断以匹配记录为准。

"""
    text+=f"""### 4.6 耗时与收益成本

{table(timings,["方法","去噪ms/张","检测ms/张","合计ms/张"])}

表5：先求每个条件内的逐图平均，噪声条件再按三个种子汇总。每种方法最终得到7组结果（1组干净输入、6组噪声类型与强度组合），表中对这7组同权平均。合计仅为滤波＋检测，不包含读图与报告生成。非局部均值搜索更多块，带来的开销需与检测收益一起判断；图像更复杂时NMS和预测数量也会改变检测耗时。

## 5. 实际难点与处理方法

### 5.1 GPU环境兼容

最初检测到的PyTorch 2.7.1+cu118能完成基础张量和卷积，却提示RTX 5070 Ti的sm120不受该构建正式支持。真实YOLO预热长时间未返回，不能由基础CUDA测试通过推出检测器可用。中断该进程后，在项目虚拟环境中安装2.7.1+cu128与对应torchvision，再完成真实图像六方法推理与小矩阵，正式实验才统一启用GPU。耗时统计从环境调整后的正式运行开始；早期挂起的原因尚不能定位到某个具体算子。

### 5.2 路径、下载与配置权限

Windows中文和空格路径下使用fromfile＋imdecode、imencode＋tofile，避免直接路径解码的不一致。工作目录固定为项目根，数据路径来自相对配置。Ultralytics默认用户目录配置写入受限，改为项目.cache/ultralytics；Matplotlib缓存也设在项目内。

直接GitHub权重下载与Git传输遇到连接重置，改用官方GitHub资产API下载并检查SHA256。COCO官方验证标注从支持Range的ZIP中只提取所需文件，图像失败重试；无法验证解码的文件不作为成功下载。

### 5.3 颜色、缩放与类别映射

OpenCV使用BGR；传NumPy数组给Ultralytics时由其内部按API约定转换，自己再交换通道会造成错误。Matplotlib展示时才BGR转RGB。真值框使用精确sx/sy，预测框使用模型映射回原处理图像的坐标，避免把letterbox尺寸和处理图坐标混用。COCO类别ID有空号，所以显式用80类别列表映射，未用cls+1。

### 5.4 评价与运行记录

直接统计预测框数量，容易忽略重复匹配、crowd区域和空预测等情况。改用COCOeval统一匹配，基础检查包含重复预测、错类别、crowd、空预测。AP保留低置信度尾部，F1单独阈值化。每图记录预测、质量、耗时与处理图哈希，每条件保存指纹；断点恢复禁止旧输入与新参数混在一起。

分阶段验证还发现：命令行浮点25.0与JSON整数25进入字符串随机键时表示不同，会造成演示与正式实验的噪声不一致。演示入口将整数强度归一为int后再传给同一核心函数，重新核对默认演示与正式配置的噪声像素相同。正式矩阵使用固定JSON整数强度，未改变核心实验输入。

彩色非局部均值最耗时，因此先完成8张小矩阵验证，再一次运行正式矩阵。报告直接读取原始记录，避免报告生成时重复检测或产生另一套预测结果。各组结果先落盘，运行被打断时只需补齐尚未完成的条件。

## 6. 结果解释、局限与总结心得

### 6.1 可以支持的结论

固定参数的滤波选择依赖噪声类型与强度，保边操作与排序操作有不同信息取舍。应同时看AP、固定阈值P/R/F1、图像质量、局部细节和耗时，而不能只看一张“更干净”的图。干净对照揭示额外处理自身的损失；成对噪声输入排除了各方法使用不同随机图像的比较偏差。

本次高斯噪声{counts["gaussian"]}/15、椒盐噪声{counts["salt_pepper"]}/15个滤波组合在平均AP上超过不去噪；其余未超过，并不等于这些方法在所有参数下无效。不同噪声分布、模型、输入分辨率或滤波参数都可能改变结果。成对AP差的三个种子原值、均值和标准差保存到analysis_summary.json，可核查收益是否对随机实现敏感。

### 6.2 局限与工业缺陷启发

128张只是COCO固定小样本，部分类别没有真值；三个种子只检验噪声随机性，不检验抽样方差。合成噪声不含工况照明、模糊、压缩、纹理变化等复杂因素；固定预训练模型没有对噪声训练。标准差不是显著性证明，也没有跨模型实验。本作业检测普通目标，并没有验证工业缺陷算法或在线学习。

对于弱划痕、小缺口等缺陷，高频可能恰好是必须保留的证据。联系DCP-SFR与UniRestore，可把“恢复清晰度”转为“保持下游证据”的研究问题：在冻结参考模型下分别评估正常漂移适应与弱缺陷得分保持，然后才考虑在线更新。若处理把重复缺陷变得像正常样本，还需独立监控参考库污染与回滚；本实验提供取舍示例，不证明该时序机制已经解决。

### 6.3 实验心得与后续改进

实现过程中，环境兼容性、细节保留和评价匹配是最需要检查的三个环节。基础CUDA测试通过后，仍需运行完整模型确认设备兼容；比较去噪结果时，应把局部图像、检测指标和真值匹配放在一起分析。保留输入、参数及逐图预测，有助于定位异常结果，也能让后续的指标核算和案例分析有据可查。

后续可在独立调参集比较窗口大小与h，在另一批图像和检测器上检验结论，再加入接近真实工况的退化。改动应使用新的输出目录，保留本次固定实验记录。

## 7. 复现入口与记录索引

- configs/full.json：固定设计及所有算法/检测参数。
- results/full/environment.json、config.json、completion.json：实际环境、指纹和114组运行摘要。
- results/full/data_manifest.json、annotations_resized.json：来源、许可与缩放真值。
- results/full/conditions：每条件汇总；records：逐图预测、TP/FP/FN、质量、时间和图像哈希。
- results/full/condition_results.csv：114行；aggregate_results.csv/json：42行均值与标准差。
- analysis_summary.json：成对AP差与案例选择结果；generate_report.py可从正式记录重建本文数字和图表。

个人PyCharm指南与练习记录在两个仓库之外的local-learning，不提交GitHub。

## 参考资料

[1] [OpenCV平滑滤波官方说明](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)。

[2] [OpenCV非局部均值官方说明](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)。

[3] [Ultralytics YOLO11官方模型文档](https://docs.ultralytics.com/models/yolo11/)，[固定发布资产](https://github.com/ultralytics/assets/releases/tag/v8.3.0)。

[4] [COCO数据下载](https://cocodataset.org/#download)，[官方bbox评价实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。

[5] [scikit-image结构相似度API](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)。

[6] 第三章课件，第10页作业要求与第29—41页相关滤波内容；作业1中DCP-SFR与UniRestore论文分析。
"""
    output.mkdir(parents=True,exist_ok=True)
    (output/"report.md").write_text(text,encoding="utf-8")
    save_json(output/"analysis_summary.json",{"completion":complete,"paired_ap_deltas":paired,
              "positive_delta_combinations":counts,"cases":cases,"figure_source":"saved full-matrix predictions; no new inference"})
    if pdf:
        render(output/"report.md",output/"report.pdf")
    print("Report regenerated from",inp)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="results/full")
    ap.add_argument("--output",default=".")
    ap.add_argument("--no-pdf",action="store_true")
    args=ap.parse_args()
    main(args.input,args.output,not args.no_pdf)
