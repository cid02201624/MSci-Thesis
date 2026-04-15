import pandas as pd
from Training_Data_Generation.Processing2 import precompute_split_to_pt_shards
import torch
import time
# from Training_Data_Generation.Sampling import write_fixed_size_split_csvs_from_allocated
# from Training_Data_Generation.Sampling import write_splits_csv

# path = write_splits_csv(
#     out_dir="splits",
#     file_name="network_segments_list.json",
#     seglen=8,
#     padding=30,
#     n_val=500_000,
#     n_test=500_000,
#     seed=2026,
#     class_probs=(1/3, 1/3, 1/3),
#     # compression="gzip",  # recommended for size
# )
# print("Wrote:", path)

# paths = write_fixed_size_split_csvs_from_allocated(
#     allocated_csv="Training_Data_Generation/splits/all_available_starts_blocksplit.csv",
#     out_dir="Training_Data_Generation/splits/precompute_1M",
#     n_train=1_000_000,
#     n_val=64,
#     n_test=64,
#     seed=2026,
# )
# print(paths)

# REMEMBER TRANSPOSE IN TO SPECTROGRAM FUNCTION

if __name__ == "__main__":
    train_df = pd.read_csv("Training_Data_Generation/splits/precompute_1M/train_1M.csv")
    # val_df   = pd.read_csv("Training_Data_Generation/splits/precompute_300k/val_30k.csv")
    # test_df  = pd.read_csv("Training_Data_Generation/splits/precompute_300k/test_30k.csv")

    max_job_retries = 5
    for attempt in range(max_job_retries):
        try:
            precompute_split_to_pt_shards(
                train_df,
                out_dir="Training_Data_Generation/pt_dataset/train",
                batch_size=64,
                num_workers=8,   # gentler on GWOSC API
                cache=True,
                x_dtype=torch.float16,
            )
            break
        except Exception as e:
            print(f"[top-level retry] attempt {attempt+1}/{max_job_retries} failed: {e}", flush=True)
            if attempt == max_job_retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
