import pandas as pd
from Training_Data_Generation.Processing2 import precompute_split_to_pt_shards
import torch
import time
from Training_Data_Generation.Processing2 import write_test_metadata_csv_from_csv

out_path = write_test_metadata_csv_from_csv(
    test_csv_path="Training_Data_Generation/splits/precompute_300k/test_30k.csv",
    out_csv="Training_Data_Generation/pt_dataset/test_sample_metadata copy.csv",
    batch_size=64,
    num_workers=4,
    cache=True,
    loader_seed=2026,
    seglen=8,
    sample_rate=4096,
    padding=30,
)

print("Wrote:", out_path)

