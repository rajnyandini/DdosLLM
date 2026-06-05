import pandas as pd
import numpy as np
import os
from scapy.all import PcapReader, IP, TCP, UDP
import math
import argparse

DATA_FOLDER = "data"
PROCESSED_FOLDER = os.path.join("scripts", "processed_data")
MAWI_FILE = "mawi_flows.csv"
PCAP_FILE = "202401011400.pcap"

def extract_mawi_test_sample(pcap_path, output_csv, max_packets=100000):
    flows = {}
    packet_count = 0
    
    print(f"Starting test: Reading first {max_packets} packets from {pcap_path}...")
    
    with PcapReader(pcap_path) as pcap_reader:
        for pkt in pcap_reader:
            packet_count += 1
            
            if packet_count % 10000 == 0:
                print(f"Progress: {packet_count} packets processed...")
            
            if packet_count > max_packets:
                print(f"Reached limit of {max_packets} packets. Stopping extraction.")
                break
            
            if not pkt.haslayer(IP):
                continue
            
            ip_layer = pkt.getlayer(IP)
            proto = ip_layer.proto
            src_port = 0
            dst_port = 0
            
            if proto == 6 and pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            elif proto == 17 and pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
            else:
                continue
            
            flow_key = (ip_layer.src, ip_layer.dst, src_port, dst_port, proto)
            pkt_len = len(pkt)
            
            if flow_key not in flows:
                flows[flow_key] = [1, pkt_len, pkt_len**2, pkt_len, pkt_len]
            else:
                stats = flows[flow_key]
                stats[0] += 1
                stats[1] += pkt_len
                stats[2] += pkt_len**2
                if pkt_len < stats[3]: stats[3] = pkt_len
                if pkt_len > stats[4]: stats[4] = pkt_len
    
    flow_data = []
    for key, s in flows.items():
        count = s[0]
        total_bytes = s[1]
        mean_len = total_bytes / count
        variance = (s[2] / count) - (mean_len ** 2)
        std_len = math.sqrt(max(0, variance)) if count > 1 else 0.0
        
        flow_data.append([
            key[2], key[3], key[4], 
            total_bytes, count,     
            mean_len, s[4], s[3],   
            std_len, 0              
        ])
    
    cols = ['src_port', 'dst_port', 'proto', 'total_bytes', 'total_pkts', 
            'mean_pkt_len', 'max_pkt_len', 'min_pkt_len', 'std_pkt_len', 'label']
    
    df = pd.DataFrame(flow_data, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"Extracted {len(df)} flows from {packet_count-1} packets.")
    print(f"CSV saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For generating csv file from pcap")
    parser.add_argument("--maxpackets", type=int, default=1_000_000, help="Max packets to read")

    args = parser.parse_args()

    extract_mawi_test_sample(os.path.join(DATA_FOLDER, PCAP_FILE), os.path.join(PROCESSED_FOLDER, MAWI_FILE), max_packets=args.maxpackets)