import re
from pathlib import Path

import pandas as pd

PATIENT_ID_RE = re.compile(r"(moyamoya_stanford_(\d{4})_\d{3})")


def extract_patient_id(text: str) -> str:
    m = PATIENT_ID_RE.search(text)
    return m.group(1) if m else None


def parse_image_type(filename: str):
    return filename[:-7] if filename.endswith(".nii.gz") else Path(filename).stem


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _read_and_normalize_labels(labels_files):
    labels_files = [str(p) for p in _as_list(labels_files)]
    if len(labels_files) == 0:
        raise ValueError("labels_files is empty")

    all_lbl = []
    for lf in labels_files:
        df_lbl_raw = pd.read_excel(lf)
        if "Patient ID" not in df_lbl_raw.columns:
            raise ValueError(f"labels file missing column 'Patient ID': {lf}")
        if "Pre-op mRS" not in df_lbl_raw.columns:
            raise ValueError(f"labels file missing column 'Pre-op mRS': {lf}")

        df_lbl = df_lbl_raw[["Patient ID", "Pre-op mRS"]].copy()
        df_lbl = df_lbl.rename(columns={"Patient ID": "patient_id", "Pre-op mRS": "label"})
        df_lbl["__source_file__"] = lf
        all_lbl.append(df_lbl)

    df_lbl = pd.concat(all_lbl, ignore_index=True).dropna(subset=["patient_id", "label"])
    df_lbl["patient_id"] = df_lbl["patient_id"].astype(str).str.strip()
    df_lbl["label"] = pd.to_numeric(df_lbl["label"], errors="coerce")

    if df_lbl["label"].isna().any():
        bad = df_lbl[df_lbl["label"].isna()][["patient_id", "label", "__source_file__"]].head(20)
        raise ValueError("Some labels cannot be parsed as numbers (Pre-op mRS):\n" + bad.to_string(index=False))
    if not (df_lbl["label"] % 1 == 0).all():
        bad = df_lbl[(df_lbl["label"] % 1 != 0)][["patient_id", "label", "__source_file__"]].head(20)
        raise ValueError("Some labels are not integers; classification expects integer classes:\n" + bad.to_string(index=False))

    df_lbl["label"] = df_lbl["label"].astype(int)

    dup = df_lbl[df_lbl.duplicated("patient_id", keep=False)].sort_values("patient_id")
    if len(dup) > 0:
        conflict = dup.groupby("patient_id")["label"].nunique()
        conflict_ids = conflict[conflict > 1].index.tolist()
        if len(conflict_ids) > 0:
            bad = dup[dup["patient_id"].isin(conflict_ids)][["patient_id", "label", "__source_file__"]].head(50)
            raise ValueError(
                "Conflicting duplicate patient IDs across label files (same patient_id has different labels). "
                "Fix the Excel files or deduplicate:\n" + bad.to_string(index=False)
            )
        df_lbl = df_lbl.sort_values(["patient_id", "__source_file__"]).drop_duplicates(subset=["patient_id"], keep="first")

    if "__source_file__" in df_lbl.columns:
        df_lbl = df_lbl.drop(columns=["__source_file__"])
    return df_lbl


def build_tables(
    image_dirs,
    labels_files,
    patients_out_csv: str,
    images_out_csv: str,
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

    if len(paths) == 0:
        raise ValueError(f"No .nii.gz found in image_dirs={image_dirs}")

    img_rows = []
    for p in paths:
        patient_id = extract_patient_id(str(p))
        image_type = parse_image_type(p.name)
        img_rows.append(
            {
                "patient_id": patient_id,
                "image_type": image_type,
                "path": str(p),
                "filename": p.name,
            }
        )
    df_img = pd.DataFrame(img_rows)

    if df_img["patient_id"].isna().any():
        bad = df_img[df_img["patient_id"].isna()][["filename", "path"]].head(20)
        raise ValueError(
            "Some files cannot parse patient_id "
            "(expected moyamoya_stanford_<YYYY>_XXX in folder name or filename):\n"
            + bad.to_string(index=False)
        )

    df_lbl = _read_and_normalize_labels(labels_files)
    image_patient_ids = set(df_img["patient_id"].tolist())
    label_patient_ids = set(df_lbl["patient_id"].tolist())
    matched_patient_ids = sorted(image_patient_ids & label_patient_ids)
    missing_labels = sorted(image_patient_ids - label_patient_ids)
    orphan_labels = sorted(label_patient_ids - image_patient_ids)

    print(f"Image files found: {len(df_img)}")
    print(f"Unique patients in images: {len(image_patient_ids)}")
    print(f"Patients in labels: {len(label_patient_ids)}")
    print(f"Matched patients: {len(matched_patient_ids)}")
    print(f"Image patients without labels: {len(missing_labels)}")
    print(f"Label patients without images: {len(orphan_labels)}")

    if len(missing_labels) > 0:
        print("\nExample image patients without labels (first 10):")
        for pid in missing_labels[:10]:
            print(pid)
    if len(orphan_labels) > 0:
        print("\nExample labels without images (first 10):")
        print(df_lbl[df_lbl["patient_id"].isin(orphan_labels[:10])][["patient_id", "label"]].to_string(index=False))

    if strict and (len(missing_labels) > 0 or len(orphan_labels) > 0):
        raise ValueError("Strict mode: mismatch between images and labels. Fix missing/orphan entries.")

    df_patients = (
        df_lbl[df_lbl["patient_id"].isin(matched_patient_ids)][["patient_id", "label"]]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    df_images = (
        df_img[df_img["patient_id"].isin(matched_patient_ids)][["patient_id", "image_type", "path"]]
        .sort_values(["patient_id", "image_type", "path"])
        .reset_index(drop=True)
    )

    Path(patients_out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(images_out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_patients.to_csv(patients_out_csv, index=False)
    df_images.to_csv(images_out_csv, index=False)
    print("\nSaved patients table to:", patients_out_csv)
    print("Saved patient-images table to:", images_out_csv)


if __name__ == "__main__":
    print(">>> build_manifest.py started")
    build_tables(
        image_dirs="/data1/yxinwang/yarong/project/data/image_data",
        labels_files=[
            "/data1/yxinwang/yarong/project/data/label/2021_mRSmoyamoya.xlsx",
            "/data1/yxinwang/yarong/project/data/label/2022_mRSmoyamoya.xlsx",
            "/data1/yxinwang/yarong/project/data/label/2023_mRSmoyamoya.xlsx",
        ],
        patients_out_csv="/data1/yxinwang/yarong/project/src/data_handler/patients.csv",
        images_out_csv="/data1/yxinwang/yarong/project/src/data_handler/patient_images.csv",
        strict=False,
        recursive=True,
    )
