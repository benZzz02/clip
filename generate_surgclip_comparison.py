#!/usr/bin/env python3
"""
用于生成与 SurgCLIP 对比结果的脚本，支持任务完整性检查
"""

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


def check_task_completeness(input_dir):
    """
    检查任务完整性
    """
    completed_tasks = []
    missing_tasks = []
    
    # 检查是否存在 summary_all.csv 文件
    summary_path = f"{input_dir}/summary_all.csv"
    if not os.path.exists(summary_path):
        print("错误：未找到评估结果文件 summary_all.csv")
        return [], list(surgclip_data.keys())
    
    # 读取 summary_all.csv
    try:
        with open(summary_path, 'r') as file:
            reader = csv.DictReader(file)
            evaluated_datasets = [row['dataset'] for row in reader]
    except Exception as e:
        print(f"读取评估结果时出错：{e}")
        return [], list(surgclip_data.keys())
    
    # 映射到任务名称
    evaluated_tasks = [dataset_mapping.get(dataset, dataset) for dataset in evaluated_datasets]
    
    # 检查哪些任务已完成
    for task in surgclip_data.keys():
        if task in evaluated_tasks:
            completed_tasks.append(task)
        else:
            missing_tasks.append(task)
    
    return completed_tasks, missing_tasks


def print_task_status(completed_tasks, missing_tasks):
    """
    打印任务状态
    """
    print("正在检查任务完整性...")
    print()
    
    print("已完成的任务：")
    for task in completed_tasks:
        print(f"- {task} ✔️")
    print()
    
    if missing_tasks:
        print("未完成的任务：")
        for task in missing_tasks:
            print(f"- {task} ❌")
        print()
        print("请完成剩余任务的评估，然后再次运行此脚本。")
        print()
        print("建议的执行命令示例：")
        
        # 反向映射任务到数据集
        reverse_mapping = {v: k for k, v in dataset_mapping.items()}
        for task in missing_tasks:
            dataset = reverse_mapping.get(task, task)
            print(f"python3 zeroshot_evaluate.py --dataset {dataset} --ckpt your_model.ckpt --output_dir {args.input_dir}")
    
    return len(missing_tasks) == 0


def generate_surgclip_comparison(input_dir, output_dir):
    """
    生成与 SurgCLIP 各个版本的对比表格
    """
    import os
    
    # 检查任务完整性
    completed_tasks, missing_tasks = check_task_completeness(input_dir)
    
    # 打印任务状态
    all_tasks_complete = print_task_status(completed_tasks, missing_tasks)
    
    # 如果任务不完整，不生成完整对比
    if not all_tasks_complete:
        return False
    
    # 读取评估结果
    eval_data = {}

    with open(f'{input_dir}/summary_all.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset = row['dataset']
            task = dataset_mapping.get(dataset, dataset)
            
            if 'Phase' in task or 'Step' in task or 'Action' in task:
                # 对于 Phase、Step、Action 任务，提取准确率和 F1 值
                accuracy = float(row['overall_accuracy']) * 100
                f1 = float(row['overall_f1_average']) * 100
                eval_data[task] = f"{accuracy:.2f} / {f1:.2f}"
            elif 'CholeT50 Triplet mAP' == task:
                # 对于 triplet 任务，提取 triplet.mAP 字段
                if row['triplet.mAP']:
                    map_score = float(row['triplet.mAP']) * 100
                    eval_data[task] = f"{map_score:.2f}"
                else:
                    eval_data[task] = "N/A"
            elif 'mAP' in task or 'Tool' in task:
                # 对于其他 mAP 任务，提取 overall_mAP
                if row['overall_mAP']:
                    map_score = float(row['overall_mAP']) * 100
                    eval_data[task] = f"{map_score:.2f}"
                else:
                    eval_data[task] = "N/A"

    # 创建对比表格
    comparison_data = []

    for task in surgclip_data.keys():
        surgclip_beta = surgclip_data[task][1]
        surgclip_full = surgclip_data[task][2]
        current_eval = eval_data.get(task, "N/A")
        
        diff_beta = "N/A"
        diff_full = "N/A"
        
        if current_eval != "N/A":
            # 解析当前评估值
            if '/' in current_eval:
                acc_curr, f1_curr = map(float, current_eval.split('/'))
            else:
                map_curr = float(current_eval)
            
            # 计算与 SurgCLIP_beta 的差异
            if surgclip_beta != "N/A":
                if '/' in surgclip_beta:
                    acc_beta, f1_beta = map(float, surgclip_beta.split('/'))
                    diff_acc = acc_curr - acc_beta
                    diff_f1 = f1_curr - f1_beta
                    diff_beta = f"{diff_acc:+.2f} Acc / {diff_f1:+.2f} F1"
                else:
                    map_beta = float(surgclip_beta)
                    diff_map = map_curr - map_beta
                    diff_beta = f"{diff_map:+.2f}"
            
            # 计算与 SurgCLIP 的差异
            if surgclip_full != "N/A":
                if '/' in surgclip_full:
                    acc_full, f1_full = map(float, surgclip_full.split('/'))
                    diff_acc = acc_curr - acc_full
                    diff_f1 = f1_curr - f1_full
                    diff_full = f"{diff_acc:+.2f} Acc / {diff_f1:+.2f} F1"
                else:
                    map_full = float(surgclip_full)
                    diff_map = map_curr - map_full
                    diff_full = f"{diff_map:+.2f}"
        
        comparison_data.append({
            "Task": task,
            "SurgCLIP_beta": surgclip_beta,
            "SurgCLIP": surgclip_full,
            "Current": current_eval,
            "Current vs SurgCLIP_beta": diff_beta,
            "Current vs SurgCLIP": diff_full
        })

    # 保存对比结果到 CSV 文件
    comparison_df = pd.DataFrame(comparison_data)
    output_path = f'{output_dir}/comparison_results_surgclip.csv'
    comparison_df.to_csv(output_path, index=False)
    
    print()
    print(f"所有任务均已完成！ ✔️")
    print()
    print("正在生成与 SurgCLIP 对比结果...")
    print()
    print(f"对比结果已保存到 {output_path}")
    print()
    print("评估结果对比：")
    print(comparison_df.to_string(index=False))
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成与 SurgCLIP 对比结果的脚本，支持任务完整性检查"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="./",
        help="评估结果输入目录 (默认: 当前目录)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./",
        help="对比结果输出目录 (默认: 当前目录)"
    )
    
    global args
    args = parser.parse_args()
    
    # 执行对比
    success = generate_surgclip_comparison(args.input_dir, args.output_dir)
    
    if not success:
        print()
        print("警告：任务未全部完成，未生成对比结果。")
        return


if __name__ == "__main__":
    import os
    main()
