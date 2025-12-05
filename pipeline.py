
import sys
import pandas as pd
from data_quality import basic_profile, print_profile

def main(csv_path):
    print("[STEP 1] Loading Data")
    df = pd.read_csv(csv_path)
    print("[STEP 2] Running Pandas Profiling")
    profile = basic_profile(df)
    print_profile(profile)
    print("[DONE] Pipeline Finished")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py data/census_sample.csv")
        sys.exit(1)
    main(sys.argv[1])
