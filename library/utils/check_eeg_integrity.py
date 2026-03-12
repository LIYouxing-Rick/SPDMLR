import os
import glob
import argparse
import sys


def check_hinss2021_integrity(data_dir: str) -> int:
    root = os.path.join(data_dir, "MNE-Hinss2021-data")
    if not os.path.isdir(root):
        print(f"[ERROR] Root data directory not found: {root}")
        return 2

    issues = []

    for subject in range(1, 16):
        zip_name = f"P{subject:02d}.zip"
        zip_path = os.path.join(root, zip_name)
        unzip_dir = f"{zip_path}.unzip"

        if not os.path.isfile(zip_path):
            issues.append(f"[MISS] {zip_name} is missing")
            continue

        if not os.path.isdir(unzip_dir):
            issues.append(f"[MISS] unzip directory not found: {unzip_dir}")
            continue

        pattern = os.path.join(unzip_dir, f"P{subject:02d}", "S?", "eeg", "alldata_*.set")
        set_files = glob.glob(pattern)
        if not set_files:
            issues.append(f"[MISS] No .set files found for P{subject:02d} under {unzip_dir}")
            continue

        for set_path in set_files:
            fdt_path = set_path[:-4] + ".fdt"
            if not os.path.isfile(fdt_path):
                issues.append(f"[MISS] .fdt missing for: {set_path}")
                continue
            try:
                size = os.path.getsize(fdt_path)
            except Exception:
                size = -1
            if size <= 0:
                issues.append(f"[BAD ] .fdt size invalid ({size} bytes): {fdt_path}")

    if issues:
        print("\nIntegrity check found issues:")
        for msg in issues:
            print(" ", msg)
        print("\nSuggested fix: re-download problematic zips and re-extract; or delete the subject's .zip and .zip.unzip folder and rerun.")
        return 1

    print("All .set/.fdt pairs look OK across subjects P01..P15.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check Hinss2021 EEGLAB .set/.fdt integrity")
    parser.add_argument("--data_dir", required=True, help="Root data_dir (contains MNE-Hinss2021-data)")
    args = parser.parse_args()
    code = check_hinss2021_integrity(args.data_dir)
    sys.exit(code)


if __name__ == "__main__":
    main()