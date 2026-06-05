import torch
import pandas as pd
import argparse
import os
from code.FlowSequentializer import FlowSequentializer

PROCESSED_FOLDER = os.path.join("scripts", "processed_data")

def generate_zero_shot_sequences(csv_path, output_pt_path, bins):
    print(f"Loading Zero-Shot CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    y = torch.tensor(df['label'].values, dtype=torch.float32)
    df_x = df.drop(columns=['label', 'timestamp', 'src_ip', 'dst_ip'], errors='ignore')
    
    # 1. Sequentialize RAW data
    x_tensor = torch.tensor(df_x.values, dtype=torch.float32)
    combined = torch.cat([x_tensor, y.unsqueeze(1)], dim=1)
    
    sequentializer = FlowSequentializer(num_bins=bins)
    sequences = sequentializer.generate_sequences(combined, mode='inference')
    
    seq_x = sequences[:, :, :-1]
    seq_y = sequences[:, :, -1:]
    
    # 2. Local Sequence Level Normalization
    seq_min = seq_x.min(dim=1, keepdim=True)[0]
    seq_max = seq_x.max(dim=1, keepdim=True)[0]
    
    seq_x_norm = (seq_x - seq_min) / (seq_max - seq_min + 1e-8)
    
    final_sequences = torch.cat([seq_x_norm, seq_y], dim=-1)
    
    torch.save(final_sequences, output_pt_path)
    print(f"Zero Shot sequences saved to {output_pt_path}")
    print(f"Total sequences generated: {final_sequences.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--bins", type=int, default=64, help="Number of bins")
    args = parser.parse_args()

    generate_zero_shot_sequences(
        csv_path=os.path.join(PROCESSED_FOLDER, "DatasetB.csv"), 
        output_pt_path=os.path.join(PROCESSED_FOLDER, "DatasetB.pt"),
        bins=args.bins
    )