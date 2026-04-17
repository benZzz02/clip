# 实现测试输出与 SurgCLIP 对比的功能

## 目标

在评估过程中，当测试输出完所有指标后，自动生成与 SurgCLIP 各个版本对比的结果文件，以便快速了解模型性能。

## 方法

### 1. 结构化 SurgCLIP 参考数据

- 创建 `surgclip_reference.csv` 文件，包含所有 SurgCLIP 版本的性能指标
- 支持 accuracy、F1 和 average_precision 等指标的对比
- 格式：`task,version,accuracy,f1,average_precision`

### 2. 修改 eval_report_utils.py

- 添加 `load_surgclip_reference` 函数，用于加载 SurgCLIP 参考数据
- 添加 `build_surgclip_comparison_table` 函数，用于构建与 SurgCLIP 各个版本的对比表格
- 添加 `export_surgclip_comparison` 函数，用于导出对比结果
- 添加数据集到任务名称的映射，以便正确匹配

### 3. 修改 export_evaluation_reports 函数

- 在 `export_evaluation_reports` 函数中，添加对 SurgCLIP 对比的支持
- 当 `sota_file` 参数不存在时，默认使用 SurgCLIP 参考数据
- 或者添加一个新的参数 `include_surgclip_comparison`，默认为 True

### 4. 修改评估脚本

- 在 `zeroshot_evaluate.py` 和 `zeroshot_evaluate_surglavi.py` 中，确保在调用 `export_evaluation_reports` 时启用 SurgCLIP 对比功能

## 实现步骤

### 第一步：创建参考数据文件

1. 创建 `surgclip_reference.csv` 文件，包含所有 SurgCLIP 版本的性能指标
2. 确保数据格式正确，包含所有任务和版本

### 第二步：修改 eval_report_utils.py

1. 导入所需的库
2. 实现 `load_surgclip_reference` 函数
3. 实现 `build_surgclip_comparison_table` 函数
4. 实现 `export_surgclip_comparison` 函数
5. 添加数据集名称映射

### 第三步：修改 export_evaluation_reports 函数

1. 修改 `export_evaluation_reports` 函数，添加对 SurgCLIP 对比的支持
2. 在函数结束前调用 `export_surgclip_comparison` 函数
3. 保存对比结果到输出目录

### 第四步：测试功能

1. 运行评估脚本，观察是否自动生成了与 SurgCLIP 对比的结果文件
2. 检查结果文件的内容是否正确
3. 确保对比结果符合预期

## 预期结果

- 每个数据集的评估结果目录中，会自动生成 `summary_{dataset}_vs_surgclip.csv` 文件
- 该文件包含与 SurgCLIP_beta 和 SurgCLIP 版本的对比结果
- 对比结果包括 accuracy、F1 和 average_precision 等指标的差距
- 结果文件格式清晰，易于阅读和理解