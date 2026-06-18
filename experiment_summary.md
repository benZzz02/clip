# 实验结果完整记录

## 实验配置

| 参数 | Stage1 | Stage3 |
|:-----|:------:|:------:|
| Vision Backbone | ConvNeXt-L (lemonfm.pth) | ConvNeXt-L (lemonfm.pth) |
| Text Model | marcobombieri/surgicberta | marcobombieri/surgicberta |
| Unfrozen Stages | **1** (stage 7) | **3** (stages 5,6,7) |
| Gradient Checkpointing | **off** | **on** |
| Learning Rate | 1e-4 | 1e-4 |
| Weight Decay | 0.02 | 0.02 |
| Optimizer | AdamW (0.9, 0.999) | AdamW (0.9, 0.999) |
| Scheduler | CosineAnnealing (T_max=total_steps) | CosineAnnealing (T_max=total_steps) |
| LR Warmup | none | none |
| Gradient Clipping | none | none |
| Epochs | 50 | 50 |
| Per GPU Batch Size | 128 | 128 |
| Accum Steps | 1 | 1 |
| GPUs | 2 | 2 |
| Number of Frames | 8 | 8 |
| Selection Pooling | xpool | xpool |
| Selection Loss Weight | 1.0 | 1.0 |
| Selection Loss Warmup | 2 zero / 2 ramp | 1 zero / 1 ramp |
| Hierarchical Consistency | 0 | 0 |
| Embed Dim | 256 | 256 |
| Image Size | 224 | 224 |

## 各Run信息

| Run | 对应脚本 | 输出目录 | 训练完成 | 分类数 |
|:----|:---------|:---------|:--------:|:------:|
| S1_run1 | run_train_frozen_vis_swanlab.sh | same_video_triplet_xpool_adapter_no_warmup_8f_run1 | 50 epochs | 12 |
| S1_run2 | run_train_frozen_vis_swanlab.sh | same_video_triplet_xpool_adapter_no_warmup_8f_run2 | 50 epochs | 12 |
| S1_run3 | run_train_frozen_vis_swanlab.sh | same_video_triplet_xpool_adapter_no_warmup_8f_run3 | 50 epochs | 12 |
| S3_run1 | run_train_stage567.sh | same_video_triplet_xpool_adapter_stage567_8f_run1 | 50 epochs | 12 |
| S3_run2 | run_train_stage567.sh | same_video_triplet_xpool_adapter_stage567_8f_run2 | 50 epochs | 12 |
| S3_run3 | run_train_stage567.sh | same_video_triplet_xpool_adapter_stage567_8f_run3 | 50 epochs | 12 |

## 所有Run完整结果 (12个数据集)

### 相位分类 (Acc/F1)

| Run | Cholec80 Phase | AutoLaparo Phase | HeiChole Phase | BernBypass70 Phase | StrasBypass70 Phase | GraSP Phase | GraSP Step | SARRARP50 Action |
|:----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **SurgCLIP_beta** | **57.98 / 39.42** | **55.72 / 45.95** | **56.95 / 44.00** | **18.30 / 15.06** | **31.24 / 26.05** | **34.77 / 27.98** | **14.15 / 11.14** | **13.94 / 7.62** |
| **SurgCLIP** | **61.29 / 50.53** | **69.14 / 56.37** | **63.84 / 55.15** | **23.90 / 19.68** | **32.37 / 30.78** | **41.49 / 34.94** | **26.28 / 16.53** | **17.42 / 7.76** |
| S1_run1 | 54.19 / 40.49 | 58.80 / 50.97 🟢 | 41.09 / 38.23 | 13.99 / 14.48 | 25.48 / 21.47 | 29.04 / 25.96 | 14.00 / 11.06 | 10.70 / 2.43 |
| S1_run2 | 49.52 / 36.35 | 48.82 / 40.29 | 53.38 / 41.41 | 21.66 / 19.06 🟢 | 31.39 / 25.24 🟢 | 32.34 / 29.56 | 13.36 / 11.35 | 11.85 / 5.46 |
| S1_run3 | 42.89 / 31.95 | 60.60 / 49.55 🟢 | 41.94 / 42.19 | 17.09 / 15.70 | 23.28 / 18.36 | 32.55 / 30.41 | 12.12 / 11.34 | 2.68 / 1.81 |
| S3_run1 | 42.28 / 30.81 | 49.27 / 42.23 | 43.87 / 36.67 | 12.53 / 13.42 | 24.34 / 23.88 | 33.70 / 30.41 | 14.65 / 10.41 🟢 | 3.85 / 3.06 |
| S3_run2 | 50.59 / 36.94 | 48.60 / 42.71 | 43.99 / 37.20 | 19.75 / 17.57 🟢 | 31.60 / 27.01 🟢 | 32.92 / 28.56 | 15.60 / 10.32 🟢 | 3.85 / 3.47 |
| S3_run3 | 52.67 / 39.76 | 49.32 / 46.82 | 46.42 / 39.90 | 18.40 / 16.74 🟢 | 29.71 / 25.41 | 32.60 / 27.88 | 15.65 / 13.09 🟢 | 2.32 / 1.09 |

🟢 = 达到或超过SurgCLIP_beta

### 工具检测 (mAP)

| Run | Cholec80 Tool (mAP) | HeiChole Tool (mAP) | GraSP Tool (mAP) |
|:----|:----:|:----:|:----:|
| **SurgCLIP_beta** | **36.77** | **31.47** | **43.06** |
| **SurgCLIP** | **40.80** | **36.79** | **45.97** |
| S1_run1 | 34.41 | 44.72 🟢 | 44.22 🟢 |
| S1_run2 | 34.81 | 41.04 🟢 | 43.84 🟢 |
| S1_run3 | 31.10 | 29.71 | 42.96 |
| S3_run1 | 30.80 | 34.91 🟢 | 41.73 |
| S3_run2 | 31.49 | 37.09 🟢 | 41.88 |
| S3_run3 | 29.88 | 37.77 🟢 | 40.91 |

🟢 = 达到或超过SurgCLIP_beta

### Triplet检测 (mAP)

| Run | CholeT50 Triplet (mAP) |
|:----|:----------------------:|
| **SurgCLIP_beta** | **4.17** |
| **SurgCLIP** | **5.28** |
| S1_run1 | 4.02 |
| S1_run2 | - |
| S1_run3 | - |
| S3_run1 | 3.91 |
| S3_run2 | 3.74 |
| S3_run3 | 3.84 |

## 超SurgCLIP_beta统计

| Run | 配置 | 超beta/总数 | 具体超beta的数据集 |
|:----|:----:|:----------:|:------------------|
| S1_run1 | 1stage, noGC, LR=1e-4 | 3/12 | AutoLaparo Phase, HeiChole Tool (mAP), GraSP Tool (mAP) |
| S1_run2 | 1stage, noGC, LR=1e-4 | 4/11 | BernBypass70 Phase, StrasBypass70 Phase, HeiChole Tool (mAP), GraSP Tool (mAP) |
| S1_run3 | 1stage, noGC, LR=1e-4 | 1/11 | AutoLaparo Phase |
| S3_run1 | 3stages, GC, LR=1e-4 | 2/12 | GraSP Step, HeiChole Tool (mAP) |
| S3_run2 | 3stages, GC, LR=1e-4 | 4/12 | BernBypass70 Phase, StrasBypass70 Phase, GraSP Step, HeiChole Tool (mAP) |
| S3_run3 | 3stages, GC, LR=1e-4 | 3/12 | BernBypass70 Phase, GraSP Step, HeiChole Tool (mAP) |

## 各run与SurgCLIP_beta差距

| Run | Cholec80P | AutoLaparo | HeiChole | Cholec80T | HeiCholeT | GraSPT |
|:----|:---------:|:----------:|:--------:|:---------:|:---------:|:------:|
| S1_run1 | -3.8 | **+3.1** | -15.9 | -2.4 | **+13.2** | **+1.2** |
| S1_run2 | -8.5 | -6.9 | -3.6 | -2.0 | **+9.6** | **+0.8** |
| S1_run3 | -15.1 | **+4.9** | -15.0 | -5.7 | -1.8 | -0.1 |
| S3_run1 | -15.7 | -6.4 | -13.1 | -6.0 | **+3.4** | -1.3 |
| S3_run2 | -7.4 | -7.1 | -13.0 | -5.3 | **+5.6** | -1.2 |
| S3_run3 | -5.3 | -6.4 | -10.5 | -6.9 | **+6.3** | -2.1 |

## 关键结论

1. **Stage1 vs Stage3基本持平**：S1_run1在主要任务Cholec80 Phase上最高（54.19），S3_run3次高（52.67）。
2. **超beta能力**：S1_run2和S3_run2均超4/11个数据集，但S1_run1在最优任务上的表现更抢眼（AutoLaparo+3.1, HeiCholeTool+13.3）。
3. **HeiCholeTool最强势**：5/6个run超过SurgCLIP_beta。
4. **AutoLaparo仅Stage1跑赢**：S1_run1(+3.1)和S1_run3(+4.9)超过beta，所有Stage3 run均低于beta。
5. **Cholec80Phase差距稳定**：所有run均低于beta，S1最佳差-3.8，S3最佳差-5.3。
6. **效率考量**：Stage1不需要gradient checkpointing，训练更快；Stage3需要GC但精度无提升。