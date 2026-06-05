import pandas as pd
import numpy as np
import os
import random
import argparse

DATA_FOLDER = "data"
PROCESSED_FOLDER = os.path.join("scripts", "processed_data")
MAWI_FILE = "mawi_flows.csv"

ATTACK_FILES = [
    "DrDos_DNS.csv", "DrDos_LDAP.csv", "DrDos_MSSQL.csv", "DrDos_NetBIOS.csv",
    "DrDos_NTP.csv", "DrDos_UDP.csv", "Syn.csv", "UDPLag.csv", "DrDos_SNMP.csv",
    "DrDos_SSDP.csv", "Portmap.csv"
]

# Dataset A (DNS, NTP, and SYN)
DatasetA = [
   "DrDoS_DNS.csv", "DrDos_NTP.csv", "Syn.csv"
]

# Dataset B (All other files)
DatasetB = [file for file in ATTACK_FILES if file not in DatasetA]

# Spoof source ips
def generate_random_ip():
    return f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"

# Generate random ips in 10 C subnet networks, this is for destination ips
def generate_carpet_bombing_dst_ip():
    base_prefix = "192.168."
    subnet_offset = random.randint(0, 9)
    host_offset = random.randint(0, 255)
    return f"{base_prefix}{subnet_offset}.{host_offset}"

def build_carpet_bombing_dataset(files, mix_ratio, total_samples, output_file):
    attack_frames = []
    cols_map = {
        'Source Port': 'src_port', 'Destination Port': 'dst_port', 
        'Protocol': 'proto', 'Total Length of Fwd Packets': 'total_bytes', 
        'Total Packets': 'total_pkts', 'Packet Length Mean': 'mean_pkt_len', 
        'Max Packet Length': 'max_pkt_len', 'Min Packet Length': 'min_pkt_len', 
        'Packet Length Std': 'std_pkt_len'
    }

    for f in files:
        path = os.path.join(DATA_FOLDER, f)
        if os.path.exists(path):
            df = pd.read_csv(path, skipinitialspace=True)

            if (args.inject_noise == False): # Essentially set all flows in attack files as attack even tho some are benign
                print("Not noise injection")
                df.columns = df.columns.str.strip()
                df = df[df['Label'] != 'BENIGN']
            
            df = df.rename(columns=cols_map)

            df = df[[c for c in cols_map.values() if c in df.columns]]
            for c in cols_map.values():
                if c not in df.columns: df[c] = 0
            attack_frames.append(df)
    
    all_attacks = pd.concat(attack_frames, ignore_index=True)
    all_attacks['label'] = 1
    
    benign_path = os.path.join(PROCESSED_FOLDER, MAWI_FILE)
    all_benign = pd.read_csv(benign_path)
    all_benign['label'] = 0
    
    num_attacks = int(total_samples * mix_ratio)
    num_benign = total_samples - num_attacks
    
    sampled_attacks = all_attacks.sample(n=min(num_attacks, len(all_attacks)), replace=True)
    sampled_benign = all_benign.sample(n=min(num_benign, len(all_benign)), replace=True)
    
    final_df = pd.concat([sampled_attacks, sampled_benign], ignore_index=True)
    
    start_ts = 1704067200 
    end_ts = 1704153600
    final_df['timestamp'] = [random.randint(start_ts, end_ts) for _ in range(len(final_df))]
    
    final_df['src_ip'] = [generate_random_ip() for _ in range(len(final_df))]
    final_df['dst_ip'] = [generate_carpet_bombing_dst_ip() for _ in range(len(final_df))]
    
    final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)
    final_df.to_csv(f"{output_file}", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For generating different datasets")
    parser.add_argument("--dataset", type=str, default="A", help="Select which dataset")
    parser.add_argument("--ratio", type=float, default=0.8, help="Mix ratio between benign and attack")
    parser.add_argument("--inject_noise", action="store_true", help="Include benign rows in attack files")

    args = parser.parse_args()

    if args.dataset == "Balanced":
        print("Generating Dataset Balanced")
        build_carpet_bombing_dataset(ATTACK_FILES, 0.5, 40_000, f"{PROCESSED_FOLDER}/Balanced.csv")

    elif args.dataset == "Imbalanced":
        print("Generating Dataset Imbalanced")
        build_carpet_bombing_dataset(ATTACK_FILES, args.ratio, 40_000, f"{PROCESSED_FOLDER}/Imbalanced.csv")

    elif args.dataset == "A":
        print("Generating Dataset A")
        build_carpet_bombing_dataset(DatasetA, 0.5, 40_000, f"{PROCESSED_FOLDER}/DatasetA.csv")

    elif args.dataset == "B":
        print("Generating Dataset B")
        build_carpet_bombing_dataset(DatasetB, 0.5, 40_000, f"{PROCESSED_FOLDER}/DatasetB.csv")