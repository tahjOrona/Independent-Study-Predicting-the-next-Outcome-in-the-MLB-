import subprocess
import sys
import os
import glob

"""
# ==============================
# MAIN PIPELINE (Run Regular Pipeline)
# ==============================
def run_pipeline(run_number):
    print(f"\n{'=' * 60}")
    print(f"Baseball Outcome Prediction Pipeline — Run {run_number}/10")
    print("=" * 60)

    print("\n[Step 1/3] Running lookup.py...")
    print("-" * 60)
    result = subprocess.run([sys.executable, "lookup.py"])
    if result.returncode != 0:
        print("❌ lookup.py failed")
        return False
    print("✅ Data collection complete")

    csv_files = glob.glob("combined_P*_H*.csv")
    if not csv_files:
        print("\n❌ Error: No combined CSV files found!")
        return False
    print(f"\nFound {len(csv_files)} combined CSV file(s)")

    print("\n[Step 2/3] Running prepare_dataset.py...")
    print("-" * 60)
    result = subprocess.run([sys.executable, "prepare_dataset.py"])
    if result.returncode != 0:
        print("❌ prepare_dataset.py failed")
        return False

    if not os.path.exists('training_dataset.csv'):
        print("\n❌ Error: training_dataset.csv was not created!")
        return False
    print("✅ Training dataset prepared")

    print("\n[Step 3/3] Running val_newNN.py...")
    print("-" * 60)
    result = subprocess.run([sys.executable, "val_newNN.py"])
    if result.returncode != 0:
        print("❌ val_newNN.py failed")
        return False

    return True
"""
# ==============================
# MAIN PIPELINE (Run UpSampling Pipeline)
# ==============================
def run_pipeline(run_number):
    print(f"\n{'=' * 60}")
    print(f"Baseball Outcome Prediction Pipeline — Run {run_number}/10")
    print("=" * 60)

    print("\n[Step 1/3] Running lookup.py...")
    print("-" * 60)
    result = subprocess.run([sys.executable, "lookup.py"])
    if result.returncode != 0:
        print("❌ lookup.py failed")
        return False
    print("✅ Data collection complete")

    csv_files = glob.glob("combined_P*_H*.csv")
    if not csv_files:
        print("\n❌ Error: No combined CSV files found!")
        return False
    print(f"\nFound {len(csv_files)} combined CSV file(s)")

    print("\n[Step 2/3] Running upsampling_prepare_dataset.py...")
    print("-" * 60)
    result = subprocess.run([sys.executable, "upsampling_prepare_dataset.py"])
    if result.returncode != 0:
        print("❌ upsampling_prepare_dataset.py failed")
        return False

    if not os.path.exists('training_dataset.csv'):
        print("\n❌ Error: training_dataset.csv was not created!")
        return False
    print("✅ Training dataset prepared")

    print("\n[Step 3/3] Running upsample_newNN.py...")
    print("-" * 60)
    result = subprocess.run([sys.executable, "upsample_newNN.py"])
    if result.returncode != 0:
        print("❌ upsample_newNN.py failed")
        return False

    return True



def main():
    results = []
    for i in range(1, 2):
        success = run_pipeline(i)
        results.append((i, "✅ Success" if success else "❌ Failed"))
        if not success:
            print(f"\n⚠️  Run {i} failed. Continuing to next run...")

    print("\n" + "=" * 60)
    print("=" * 60)
    for run_num, status in results:
        print(f"  Run {run_num:>2}: {status}")
    successes = sum(1 for _, s in results if "Success" in s)
    print(f"\nCompleted: {successes}/ 1 runs successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
