import pandas as pd
import torch

def extract_stats_from_existing_csv(csv_path, output_stats_path):
    df = pd.read_csv(csv_path)
    df_x = df.drop(columns=['label', 'timestamp', 'src_ip', 'dst_ip'], errors='ignore')
    
    train_stats = {
        'min': df_x.min().to_dict(),
        'max': df_x.max().to_dict()
    }
    
    torch.save(train_stats, output_stats_path)
    print(f"Success! Saved training stats to {output_stats_path}")
    print("Min values:", train_stats['min'])
    print("Max values:", train_stats['max'])

if __name__ == "__main__":
    extract_stats_from_existing_csv('../scripts/processed_data/carpet_bombing_dataset.csv', '../scripts/processed_data/train_normalization_stats.pth')