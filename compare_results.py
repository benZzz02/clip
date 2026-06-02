import csv
import pandas as pd

# 用户提供的 SurgCLIP 数据
surgclip_data = {
    "Cholec80 Phase": ["41.72 / 29.44", "57.98 / 39.42", "61.29 / 50.53", "-16.26 Acc / -9.98 F1", "-19.57 Acc / -21.09 F1"],
    "AutoLaparo Phase": ["55.91 / 49.47", "55.72 / 45.95", "69.14 / 56.37", "+0.19 Acc / +3.52 F1", "-13.23 Acc / -6.90 F1"],
    "StrasBypass70 Phase": ["31.94 / 25.84", "31.24 / 26.05", "32.37 / 30.78", "+0.70 Acc / -0.21 F1", "-0.43 Acc / -4.94 F1"],
    "HeiChole Phase": ["31.8 / 23.84", "56.95 / 44.00", "63.84 / 55.15", "-25.15 Acc / -20.16 F1", "-32.04 Acc / -31.31 F1"],
    "BernBypass70 Phase": ["15.75 / 15.77", "18.30 / 15.06", "23.90 / 19.68", "-2.55 Acc / +0.71 F1", "-8.15 Acc / -3.91 F1"],
    "GraSP Phase": ["35.43 / 31.85", "34.77 / 27.98", "41.49 / 34.94", "+0.66 Acc / +3.87 F1", "-6.06 Acc / -3.09 F1"],
    "GraSP Step": ["24.41 / 15.61", "14.15 / 11.14", "26.28 / 16.53", "+10.26 Acc / +4.47 F1", "-1.87 Acc / -0.92 F1"],
    "SARRARP50 Action": ["29.46 / 11.32", "13.94 / 7.62", "17.42 / 7.76", "+15.52 Acc / +3.70 F1", "+12.04 Acc / +3.56 F1"],
    "CholeT50 Triplet mAP": ["3.97", "4.17", "5.28", "-0.20", "-1.31"],
    "Cholec80 Tool mAP": ["33.05", "36.77", "40.80", "-3.72", "-7.75"],
    "HeiChole Tool mAP": ["28.15", "31.47", "36.79", "-3.32", "-8.64"],
    "GraSP Tool mAP": ["48.08", "43.06", "45.97", "+5.02", "+2.11"]
}

# 数据集名称映射
dataset_mapping = {
    "cholec80_phase": "Cholec80 Phase",
    "autolaparo_phase": "AutoLaparo Phase",
    "stras_bypass70_phase": "StrasBypass70 Phase",
    "heichole_phase": "HeiChole Phase",
    "bern_bypass70_phase": "BernBypass70 Phase",
    "grasp_phase": "GraSP Phase",
    "grasp_step": "GraSP Step",
    "sarrarp50_phase": "SARRARP50 Action",
    "cholect50_triplet": "CholeT50 Triplet mAP",
    "cholec80_instrument": "Cholec80 Tool mAP",
    "heichole_instrument": "HeiChole Tool mAP",
    "grasp_instrument": "GraSP Tool mAP"
}

# 读取 eval_4.7_epoch_20 的结果
eval_47_epoch20_data = {}

with open('/mnt/mydisk/CLIP/eval_4.7_epoch_20/summary_all.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        dataset = row['dataset']
        task = dataset_mapping.get(dataset, dataset)
        
        if 'Phase' in task or 'Step' in task or 'Action' in task:
            # 对于 Phase、Step、Action 任务，提取准确率和 F1 值
            accuracy = float(row['overall_accuracy']) * 100
            f1 = float(row['overall_f1_average']) * 100
            eval_47_epoch20_data[task] = f"{accuracy:.2f} / {f1:.2f}"
        elif 'mAP' in task or 'Tool' in task:
            # 对于 mAP 任务，提取 overall_mAP
            if row['overall_mAP']:
                map_score = float(row['overall_mAP']) * 100
                eval_47_epoch20_data[task] = f"{map_score:.2f}"
            else:
                eval_47_epoch20_data[task] = "N/A"

# 创建对比表格
comparison_data = []

for task in surgclip_data.keys():
    # 获取各版本数据
    eval_43_epoch50 = surgclip_data[task][0]
    surgclip_beta = surgclip_data[task][1]
    surgclip_full = surgclip_data[task][2]
    eval_47_epoch20 = eval_47_epoch20_data.get(task, "N/A")
    
    # 计算差异
    diff_47_vs_43 = "N/A"
    diff_47_vs_beta = "N/A"
    diff_47_vs_full = "N/A"
    
    if eval_47_epoch20 != "N/A":
        # 解析 eval_4.7_epoch_20 的值
        if '/' in eval_47_epoch20:
            acc_47, f1_47 = map(float, eval_47_epoch20.split('/'))
        else:
            map_47 = float(eval_47_epoch20)
        
        # 计算与 eval_4.3_epoch_50 的差异
        if eval_43_epoch50 != "N/A":
            if '/' in eval_43_epoch50:
                acc_43, f1_43 = map(float, eval_43_epoch50.split('/'))
                diff_acc = acc_47 - acc_43
                diff_f1 = f1_47 - f1_43
                diff_47_vs_43 = f"{diff_acc:+.2f} Acc / {diff_f1:+.2f} F1"
            else:
                map_43 = float(eval_43_epoch50)
                diff_map = map_47 - map_43
                diff_47_vs_43 = f"{diff_map:+.2f}"
        
        # 计算与 SurgCLIP_beta 的差异
        if surgclip_beta != "N/A":
            if '/' in surgclip_beta:
                acc_beta, f1_beta = map(float, surgclip_beta.split('/'))
                diff_acc = acc_47 - acc_beta
                diff_f1 = f1_47 - f1_beta
                diff_47_vs_beta = f"{diff_acc:+.2f} Acc / {diff_f1:+.2f} F1"
            else:
                map_beta = float(surgclip_beta)
                diff_map = map_47 - map_beta
                diff_47_vs_beta = f"{diff_map:+.2f}"
        
        # 计算与 SurgCLIP 的差异
        if surgclip_full != "N/A":
            if '/' in surgclip_full:
                acc_full, f1_full = map(float, surgclip_full.split('/'))
                diff_acc = acc_47 - acc_full
                diff_f1 = f1_47 - f1_full
                diff_47_vs_full = f"{diff_acc:+.2f} Acc / {diff_f1:+.2f} F1"
            else:
                map_full = float(surgclip_full)
                diff_map = map_47 - map_full
                diff_47_vs_full = f"{diff_map:+.2f}"
    
    comparison_data.append({
        "Task": task,
        "eval_4.3_epoch_50": eval_43_epoch50,
        "SurgCLIP_beta": surgclip_beta,
        "SurgCLIP": surgclip_full,
        "eval_4.7_epoch_20": eval_47_epoch20,
        "eval_4.7_vs_4.3": diff_47_vs_43,
        "eval_4.7_vs_beta": diff_47_vs_beta,
        "eval_4.7_vs_full": diff_47_vs_full
    })

# 保存对比结果到 CSV 文件
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('/mnt/mydisk/CLIP/eval_4.7_epoch_20/comparison_results.csv', index=False)
print("对比结果已保存到 /mnt/mydisk/CLIP/eval_4.7_epoch_20/comparison_results.csv")

# 打印对比结果
print("\n评估结果对比：")
print(comparison_df.to_string(index=False))
