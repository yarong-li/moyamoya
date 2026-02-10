#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# 配置路径
source_base = Path("/data/mosszhao/")
target_dir = Path("/data1/yxinwang/yarong/project/data")

# 文件路径模式
file_pattern = "derived/pre_surgery_yes_diamox/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_pre_diamox_standard_nonlin.nii.gz"

# 创建目标文件夹
target_dir.mkdir(parents=True, exist_ok=True)

# 遍历患者编号 001-066
copied_files = []
missing_files = []

for i in range(1, 67):
    patient_id = f"moyamoya_stanford_2023_{i:03d}"
    source_file = source_base / f"moyamoya_2023_nifti/{patient_id}" / file_pattern
    
    if source_file.exists():
        # 重命名以包含患者ID，避免覆盖
        target_file = target_dir / f"{patient_id}_CBF_Single_Delay_Pre_Diamox.nii.gz"
        shutil.copy2(source_file, target_file)
        copied_files.append(patient_id)
        print(f"✅ 已复制: {patient_id}")
    else:
        missing_files.append(patient_id)
        print(f"⚠️  未找到: {patient_id}")

# 总结报告
print(f"\n{'='*50}")
print(f"📊 复制完成！")
print(f"   成功: {len(copied_files)} 个文件")
print(f"   缺失: {len(missing_files)} 个文件")
if missing_files:
    print(f"\n缺失的患者ID: {', '.join(missing_files)}")

# 60 patients data loaded