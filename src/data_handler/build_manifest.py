import re
from pathlib import Path
import pandas as pd

# 从文件名/路径提取：moyamoya_stanford_2023_001（支持 4 位年份）
KEY_RE = re.compile(r"(moyamoya_stanford_(\d{4})_\d{3})")

def extract_key(filename: str) -> str:
    m = KEY_RE.search(filename)
    return m.group(1) if m else None

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _read_and_normalize_labels(labels_files):
    """
    Read one or more Excel label files and return a normalized dataframe with columns:
      - key (str)
      - label (int)
    """
    labels_files = [str(p) for p in _as_list(labels_files)]
    if len(labels_files) == 0:
        raise ValueError("labels_files is empty")

    all_lbl = []
    for lf in labels_files:
        df_lbl_raw = pd.read_excel(lf)

        # 选取并重命名列：Patient ID -> key, Pre-op mRS -> label
        if "Patient ID" not in df_lbl_raw.columns:
            raise ValueError(f"labels file missing column 'Patient ID': {lf}")
        if "Pre-op mRS" not in df_lbl_raw.columns:
            raise ValueError(f"labels file missing column 'Pre-op mRS': {lf}")

        df_lbl = df_lbl_raw[["Patient ID", "Pre-op mRS"]].copy()
        df_lbl = df_lbl.rename(columns={"Patient ID": "key", "Pre-op mRS": "label"})
        df_lbl["__source_file__"] = lf
        all_lbl.append(df_lbl)

    df_lbl = pd.concat(all_lbl, ignore_index=True)
    df_lbl = df_lbl.dropna(subset=["key", "label"])

    # key 规范化：去空格、转字符串
    df_lbl["key"] = df_lbl["key"].astype(str).str.strip()

    # label 处理：转 numeric，再转 int（分类用）
    df_lbl["label"] = pd.to_numeric(df_lbl["label"], errors="coerce")
    if df_lbl["label"].isna().any():
        bad = df_lbl[df_lbl["label"].isna()][["key", "label", "__source_file__"]].head(20)
        raise ValueError("Some labels cannot be parsed as numbers (Pre-op mRS):\n" + bad.to_string(index=False))

    # 如果 mRS 是 0,1,2,3...，强制转 int
    if not (df_lbl["label"] % 1 == 0).all():
        bad = df_lbl[(df_lbl["label"] % 1 != 0)][["key", "label", "__source_file__"]].head(20)
        raise ValueError("Some labels are not integers; classification expects integer classes:\n" + bad.to_string(index=False))

    df_lbl["label"] = df_lbl["label"].astype(int)

    # 合并多个文件时：允许重复 key 但必须一致（否则报错）
    dup = df_lbl[df_lbl.duplicated("key", keep=False)].sort_values("key")
    if len(dup) > 0:
        conflict = dup.groupby("key")["label"].nunique()
        conflict_keys = conflict[conflict > 1].index.tolist()
        if len(conflict_keys) > 0:
            bad = dup[dup["key"].isin(conflict_keys)][["key", "label", "__source_file__"]].head(50)
            raise ValueError(
                "Conflicting duplicate keys across label files (same key has different labels). "
                "Fix the Excel files or deduplicate:\n" + bad.to_string(index=False)
            )

        # same key, same label -> keep first occurrence
        df_lbl = df_lbl.sort_values(["key", "__source_file__"]).drop_duplicates(subset=["key"], keep="first")
    else:
        df_lbl = df_lbl.drop(columns=["__source_file__"])

    # 清理辅助列
    if "__source_file__" in df_lbl.columns:
        df_lbl = df_lbl.drop(columns=["__source_file__"])
    return df_lbl

def build_manifest(
    image_dirs,
    labels_files,
    out_csv: str,
    strict: bool = True,
    recursive: bool = True,
):
    image_dirs = [Path(p) for p in _as_list(image_dirs)]
    if len(image_dirs) == 0:
        raise ValueError("image_dirs is empty")

    paths = []
    for d in image_dirs:
        if not d.exists():
            raise ValueError(f"image_dir does not exist: {d}")
        if recursive:
            paths.extend(d.rglob("*.nii.gz"))
        else:
            paths.extend(d.glob("*.nii.gz"))
    paths = sorted({p.resolve() for p in paths})

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
        raise ValueError(f"No .nii.gz found in image_dirs={image_dirs}")

    # 无法解析 key 的文件
    if df_img["key"].isna().any():
        bad = df_img[df_img["key"].isna()][["filename"]]
        msg = (
            "Some filenames cannot parse key "
            "(expected moyamoya_stanford_<YYYY>_XXX somewhere in the filename):\n"
            + bad.to_string(index=False)
        )
        raise ValueError(msg)

    # 图片 key 必须唯一
    dup_img = df_img[df_img.duplicated("key", keep=False)].sort_values("key")
    if len(dup_img) > 0:
        raise ValueError("Duplicate image keys found:\n" + dup_img[["key", "filename"]].to_string(index=False))

    # ---------- 2) labels table ----------
    df_lbl = _read_and_normalize_labels(labels_files)

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
        # 方式 A：给多个年份目录（推荐，最清晰）
        image_dirs=[
            "/data1/yxinwang/yarong/project/data/image_data/2021",
            "/data1/yxinwang/yarong/project/data/image_data/2022",
            "/data1/yxinwang/yarong/project/data/image_data/2023",
        ],
        # 方式 B：给一个总目录，recursive=True 会递归找到所有 .nii.gz
        # image_dirs="/data1/yxinwang/yarong/project/data/image_data",
        labels_files=[
            "/data1/yxinwang/yarong/project/data/label/2021_mRSmoyamoya.xlsx",
            "/data1/yxinwang/yarong/project/data/label/2022_mRSmoyamoya.xlsx",
            "/data1/yxinwang/yarong/project/data/label/2023_mRSmoyamoya.xlsx",
        ],
        out_csv="/data1/yxinwang/yarong/project/src/data/manifest_new.csv",
        strict=False,
        recursive=True,
    )
