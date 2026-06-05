import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from model import DoLLM

def run_zero_shot_inference(model_path, sequences_path):
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = DoLLM() 
    
    model.flow_tokenizer.to(device)
    model.classification_projection.to(device)
    
    print("Loading weights from checkpoint to CPU...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model.flow_tokenizer.load_state_dict(checkpoint['tokenizer_state'])
    model.classification_projection.load_state_dict(checkpoint['projection_state'])
    
    model.eval()

    print(f"Loading sequences from {sequences_path}...")
    sequences = torch.load(sequences_path, map_location='cpu')
    
    all_preds = []
    all_targets = []
    
    batch_size = 16 
    num_sequences = sequences.size(0)

    print(f"Running inference on {num_sequences} sequences in batches...")
    
    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            batch = sequences[i : i + batch_size].to(device)
            
            x_batch = batch[:, :, :-1]
            
            y_batch = batch[:, :, -1].reshape(-1).long()
            
            logits = model(x_batch) 
            
            preds = torch.argmax(logits, dim=-1).view(-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
            del batch, x_batch, logits
            if i % 50 == 0:
                torch.cuda.empty_cache()
    
    report = classification_report(all_targets, all_preds, target_names=['Benign', 'Attack'])
    print(report)
    print(f"Final Zero Shot F1 Score: {f1_score(all_targets, all_preds):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(f"True Benign: {cm[0][0]} | False Attack: {cm[0][1]}")
    print(f"False Benign: {cm[1][0]} | True Attack: {cm[1][1]}")

if __name__ == "__main__":
    MODEL_CHECKPOINT = "/content/drive/MyDrive/dollm_best_model.pth"
    ZERO_SHOT_DATA = "processed_data/zero_shot_sequences.pt"
    
    run_zero_shot_inference(MODEL_CHECKPOINT, ZERO_SHOT_DATA)