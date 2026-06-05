import torch
import pandas as pd
import random
import argparse
import os

PROCESSED_FOLDER = os.path.join("scripts", "processed_data")

class FlowSequentializer:
    def __init__(self, num_bins=64, training_samples=15000):
        self.num_bins = num_bins
        self.training_samples = training_samples

    def _sort_flows(self, flows):
        indices = torch.sort(flows[:, 1])[1] # dst_port
        flows = flows[indices]
        indices = torch.sort(flows[:, 0], stable=True)[1] # src_port
        flows = flows[indices]
        indices = torch.sort(flows[:, 2], stable=True)[1] # proto
        flows = flows[indices]
        indices = torch.sort(flows[:, 4], stable=True)[1] # total_pkts
        flows = flows[indices]
        indices = torch.sort(flows[:, 5], stable=True)[1] # mean_pkt_len
        return flows

    def _create_flow_matrix(self, flows):
        F = flows.shape[0]
        N = self.num_bins
        q = F // N
        r = F % N
        m_max = q + (1 if r > 0 else 0)
        
        bins = []
        start_idx = 0
        for i in range(N):
            size = q + 1 if i < r else q
            current_bin = flows[start_idx : start_idx + size]
            
            # Duplicate the last flow if the bin is smaller than m_max
            if size < m_max and size > 0:
                padding = current_bin[-1].unsqueeze(0).repeat(m_max - size, 1)
                current_bin = torch.cat([current_bin, padding], dim=0)
            elif size == 0:
                current_bin = torch.zeros((m_max, flows.shape[1]), device=flows.device) # F < N
                
            bins.append(current_bin)
            start_idx += size
            
        return torch.stack(bins) # Shape: (num_bins, m_max, num_features)

    def generate_sequences(self, flows, mode='inference'):
        sorted_flows = self._sort_flows(flows)
        flow_matrix = self._create_flow_matrix(sorted_flows)
        
        if mode == 'inference':
            # Vertical selection, Transpose from (N, m, features) to (m, N, features)
            sequences = flow_matrix.transpose(0, 1)
            return sequences
        
        elif mode == 'training':
            # Limited sequences + Random sampling
            standard_seqs = flow_matrix.transpose(0, 1)
            
            random_seqs = []
            num_to_sample = max(0, self.training_samples - standard_seqs.shape[0])
            
            # Shape of flow_matrix: (64, m_max, num_features)
            m_max = flow_matrix.shape[1]
            for _ in range(num_to_sample):
                # Randomly sample one flow index from each bin
                indices = [random.randint(0, m_max - 1) for _ in range(self.num_bins)]
                seq = torch.stack([flow_matrix[i, idx] for i, idx in enumerate(indices)])
                random_seqs.append(seq)
            
            if random_seqs:
                all_training_seqs = torch.cat([standard_seqs, torch.stack(random_seqs)], dim=0)
                return all_training_seqs
            return standard_seqs

def save_sequences_to_pt(csv_path, output_pt_path, bins, mode='training'):
    df = pd.read_csv(csv_path)
    
    y = torch.tensor(df['label'].values, dtype=torch.float32)
    df_x = df.drop(columns=['label', 'timestamp', 'src_ip', 'dst_ip'], errors='ignore')
    
    # 1. Sequentialize RAW data first (No global normalization)
    x_tensor = torch.tensor(df_x.values, dtype=torch.float32)
    combined = torch.cat([x_tensor, y.unsqueeze(1)], dim=1)
    
    sequentializer = FlowSequentializer(num_bins=bins, training_samples=15000)
    sequences = sequentializer.generate_sequences(combined, mode=mode)
    
    # sequences shape is (num_seqs, 64, num_features + 1)
    seq_x = sequences[:, :, :-1]
    seq_y = sequences[:, :, -1:]
    
    # 2. Local Sequence Level Normalization
    seq_min = seq_x.min(dim=1, keepdim=True)[0]
    seq_max = seq_x.max(dim=1, keepdim=True)[0]
    
    seq_x_norm = (seq_x - seq_min) / (seq_max - seq_min + 1e-8)
    
    # 3. Recombine and shuffle
    final_sequences = torch.cat([seq_x_norm, seq_y], dim=-1)
    
    indices = torch.randperm(final_sequences.size(0))
    final_sequences = final_sequences[indices]
    
    torch.save(final_sequences, output_pt_path)
    print(f"Saved {final_sequences.shape[0]} locally-normalized sequences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--bins", type=int, default=64, help="Number of bins")

    args = parser.parse_args()

    save_sequences_to_pt(
        os.path.join(PROCESSED_FOLDER, 'DatasetA.csv'), 
        os.path.join(PROCESSED_FOLDER, 'DatasetA.pt'),
        bins=args.bins,
        mode='training'
        )
