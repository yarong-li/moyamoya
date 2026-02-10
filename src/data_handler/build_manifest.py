import re
from pathlib import Path
import pandas as pd

# 从文件名提取：moyamoya_stanford_2023_001
KEY_RE = re.compile(r"(moyamoya_stanford_2023_\d{3})")

def extract_key(filename: str) -> str:
    m = KEY_RE.search(filename)
    return m.group(1) if m else None

def build_manifest(
    image_dir: str,
    labels_file: str,
    out_csv: str,
    strict: bool = True,
):
    image_dir = Path(image_dir)
    paths = sorted(image_dir.glob("*.nii.gz"))

    # ---------- 1) images table ----------
    img_rows = []
    for p in paths:
        key = extract_key(p.name)
        img_rows.append({
            "key": key,
            "path": str(p.resolve()),
            "filename": p.name
        })
    df_img = pd.DataFrame(img_rows)

    if df_img.empty:
        raise ValueError(f"No .nii.gz found in {image_dir}")

    # 无法解析 key 的文件
    if df_img["key"].isna().any():
        bad = df_img[df_img["key"].isna()][["filename"]]
        msg = "Some filenames cannot parse key (expected moyamoya_stanford_2023_XXX):\n" + bad.to_string(index=False)
        raise ValueError(msg)

    # 图片 key 必须唯一
    dup_img = df_img[df_img.duplicated("key", keep=False)].sort_values("key")
    if len(dup_img) > 0:
        raise ValueError("Duplicate image keys found:\n" + dup_img[["key", "filename"]].to_string(index=False))

    # ---------- 2) labels table ----------
    df_lbl_raw = pd.read_excel(labels_file)

    # 选取并重命名列：Patient ID -> key, Pre-op mRS -> label
    if "Patient ID" not in df_lbl_raw.columns:
        raise ValueError("labels_csv missing column: 'Patient ID'")
    if "Pre-op mRS" not in df_lbl_raw.columns:
        raise ValueError("labels_csv missing column: 'Pre-op mRS'")

    df_lbl = df_lbl_raw[["Patient ID", "Pre-op mRS"]].copy()
    df_lbl = df_lbl.rename(columns={"Patient ID": "key", "Pre-op mRS": "label"})
    df_lbl = df_lbl.dropna(subset=["key", "label"])

    # key 规范化：去空格、转字符串
    df_lbl["key"] = df_lbl["key"].astype(str).str.strip()

    # label 处理：转 numeric，再转 int（分类用）
    df_lbl["label"] = pd.to_numeric(df_lbl["label"], errors="coerce")
    if df_lbl["label"].isna().any():
        bad = df_lbl[df_lbl["label"].isna()][["key", "label"]].head(20)
        raise ValueError("Some labels cannot be parsed as numbers (Pre-op mRS):\n" + bad.to_string(index=False))

    # 如果 mRS 是 0,1,2,3...，强制转 int
    # （如果你未来要把它当回归，这里可以改成 float32）
    if not (df_lbl["label"] % 1 == 0).all():
        bad = df_lbl[(df_lbl["label"] % 1 != 0)][["key", "label"]].head(20)
        raise ValueError("Some labels are not integers; classification expects integer classes:\n" + bad.to_string(index=False))

    df_lbl["label"] = df_lbl["label"].astype(int)

    # 标签 key 必须唯一（一个 key 一个 label）
    dup_lbl = df_lbl[df_lbl.duplicated("key", keep=False)].sort_values("key")
    if len(dup_lbl) > 0:
        raise ValueError("Duplicate label keys found in labels_csv:\n" + dup_lbl[["key", "label"]].to_string(index=False))

    # ---------- 3) merge ----------
    df = df_img.merge(df_lbl, on="key", how="inner")

    missing_labels = df_img[~df_img["key"].isin(df["key"])].sort_values("key")
    orphan_labels  = df_lbl[~df_lbl["key"].isin(df["key"])].sort_values("key")

    print(f"Images found: {len(df_img)}")
    print(f"Labels found: {len(df_lbl)}")
    print(f"Matched samples: {len(df)}")
    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(orphan_labels)}")

    if len(missing_labels) > 0:
        print("\nExample images without labels (first 10):")
        print(missing_labels.head(10)[["key", "filename"]].to_string(index=False))

    if len(orphan_labels) > 0:
        print("\nExample labels without images (first 10):")
        print(orphan_labels.head(10)[["key", "label"]].to_string(index=False))

    if strict and (len(missing_labels) > 0 or len(orphan_labels) > 0):
        raise ValueError("Strict mode: mismatch between images and labels. Fix missing/orphan entries.")

    # 输出 manifest
    df = df.sort_values("key").reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\nSaved manifest to:", out_csv)


if __name__ == "__main__":
    print(">>> build_manifest.py started")
    build_manifest(
        image_dir="/data1/yxinwang/yarong/project/data/image_data",        # TODO: 改成你的 nii.gz 所在目录
        labels_file="/data1/yxinwang/yarong/project/data/label/2023_mRSmoyamoya.xlsx",   # TODO: 改成你的 label csv 路径
        out_csv="/data1/yxinwang/yarong/project/src/data/manifest.csv",
        strict=False
    )
