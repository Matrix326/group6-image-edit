"""
# https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered
# https://huggingface.co/datasets/osunlp/MagicBrush
python process_HF.py
"""

# change the original dataset file format into .arrow file -> InstructPix2Pix + MagicBrush
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
import glob
import os
import argparse

# Define a generator function that loads Parquet files one by one and converts them into a dataset
def parquet_to_dataset_generator(file_paths):
    index = 0
    for file_path in file_paths:
        print('Number:', index)
        df = pd.read_parquet(file_path)
        dataset = Dataset.from_pandas(df)
        index = index + 1
        yield dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=str, default=None)
    parser.add_argument("--target-path", type=str, default=None)

    # same for MagicBrush
    parquet_filepaths = glob.glob(args.source_path+'/*.parquet')
    parquet_datasets = list(parquet_to_dataset_generator(parquet_filepaths))
    parquet_merged_dataset = concatenate_datasets(parquet_datasets)

    # load
    HF_path = args.target_path
    parquet_merged_dataset.save_to_disk(HF_path)
    # HF_path = load_from_disk(HF_path)
