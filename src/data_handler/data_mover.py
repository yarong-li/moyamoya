#!/usr/bin/env python3
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# 配置路径
source_base = Path("/data/mosszhao/")
target_base = Path("/data1/yxinwang/yarong/project/data/image_data")

# 每年 patient_id 的索引范围（含边界）
year_patient_ranges: Dict[int, Tuple[int, int]] = {
    2021: (1, 67),
    2022: (1, 72),
    2023: (1, 66),
}

# 每个患者需要拷贝的文件：
# (源文件相对路径, 目标文件名)
file_specs: List[Tuple[str, str]] = [
    (
        "derived/pre_surgery_yes_diamox/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz",
        "CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz",
    ),
    (
        "derived/pre_surgery_yes_diamox/anat/fsl_anat_output_t1/T1_to_MNI_nonlin.nii.gz",
        "T1_to_MNI_nonlin.nii.gz",
    ),
]

copied_count = 0
missing_count = 0
copied_patients = set()
patients_with_missing = set()

for year, (start_idx, end_idx) in year_patient_ranges.items():
    year_source_root = source_base / f"moyamoya_{year}_nifti"
    year_target_root = target_base / str(year)
    year_target_root.mkdir(parents=True, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        patient_id = f"moyamoya_stanford_{year}_{i:03d}"
        patient_source_root = year_source_root / patient_id
        patient_target_root = year_target_root / patient_id
        patient_target_root.mkdir(parents=True, exist_ok=True)

        for rel_path, target_name in file_specs:
            source_file = patient_source_root / rel_path
            target_file = patient_target_root / target_name
            

            if source_file.exists():
                shutil.copy2(source_file, target_file)
                copied_count += 1
                copied_patients.add(patient_id)
                print(f"✅ 已复制: {patient_id}/{target_name}")
            else:
                missing_count += 1
                patients_with_missing.add(patient_id)
                print(f"⚠️  未找到: {patient_id}/{target_name}")

# 总结报告
print(f"\n{'='*50}")
print(f"📊 复制完成！")
print(f"   成功文件数: {copied_count}")
print(f"   缺失文件数: {missing_count}")
print(f"   至少有 1 个文件成功的患者数: {len(copied_patients)}")
print(f"   有缺失文件的患者数: {len(patients_with_missing)}")