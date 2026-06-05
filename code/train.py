import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import DoLLM, DoLLMConv
from utils.regularizations import apply_feature_jitter, supervised_info_nce_loss
import os
import gc

def train_dollm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.load("processed_data/sequences.pt")
    X = data[:, :, :-1]
    y = data[:, :, -1].long()
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = DoLLM().to(device)
    
    optimizer = optim.Adam([
        {'params': model.flow_tokenizer.parameters()},
        {'params': model.classification_projection.parameters()}
    ], lr=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            
            logits = logits.view(-1, 2)
            batch_y = batch_y.view(-1)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                
                preds = torch.argmax(logits, dim=-1).view(-1)
                targets = batch_y.view(-1)
                
                tp += ((preds == 1) & (targets == 1)).sum().item()
                fp += ((preds == 1) & (targets == 0)).sum().item()
                fn += ((preds == 0) & (targets == 1)).sum().item()
                tn += ((preds == 0) & (targets == 0)).sum().item()
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"Epoch {epoch+1}/20 | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'tokenizer_state': model.flow_tokenizer.state_dict(),
                'projection_state': model.classification_projection.state_dict()
            }, "/content/drive/MyDrive/dollm_best_model.pth")

def train_dollmconv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dtype = torch.bfloat16
    save_path = "/content/drive/MyDrive/"
    
    gc.collect()
    torch.cuda.empty_cache()

    data = torch.load("processed_data/DatasetA.pt", map_location='cpu')
    X, y = data[:, :, :-1], data[:, :, -1].long()
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = DoLLMConv()
    model = model.to(device).to(current_dtype)
    
    loss_weights = torch.tensor([3.0, 1.0], device=device).to(current_dtype)
    criterion_ce = nn.CrossEntropyLoss(weight=loss_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=1e-4)

    best_f1 = 0.0

    print("Starting Training...")
    for epoch in range(15):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = apply_feature_jitter(batch_x, sigma=0.03).to(device).to(current_dtype)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits, hidden_states = model(batch_x, return_hidden=True)
            
            loss_ce = criterion_ce(logits.view(-1, 2), batch_y.view(-1))
            loss_nce = supervised_info_nce_loss(hidden_states.view(-1, 8192), batch_y.view(-1))
            
            loss = loss_ce + 0.1 * loss_nce
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).to(current_dtype)
                batch_y = batch_y.to(device).view(-1)
                
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=-1).view(-1)
                targets = batch_y

                tp += ((preds == 1) & (targets == 1)).sum().item()
                fp += ((preds == 1) & (targets == 0)).sum().item()
                fn += ((preds == 0) & (targets == 1)).sum().item()
                tn += ((preds == 0) & (targets == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | F1: {f1:.4f}")

        checkpoint_name = f"new_data_dollm_sota_epoch_{epoch+1}.pth"
        torch.save({
            'tokenizer_state': model.flow_tokenizer.state_dict(),
            'tokenizer_proj_state': model.tokenizer_projection.state_dict(),
            'attention_pool_state': model.attention_pool.state_dict(),
            'classification_head_state': model.classification_head.state_dict(),
        }, os.path.join(save_path, checkpoint_name))
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_dollm()
