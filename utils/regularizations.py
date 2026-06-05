import torch
import torch.nn.functional as F

def apply_feature_jitter(x, sigma=0.03):
    if sigma == 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise

def supervised_info_nce_loss(embeddings, labels, temperature=0.07):
    embeddings = F.normalize(embeddings.to(torch.float32), dim=1)
    
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    
    logits_mask = torch.scatter(
        torch.ones_like(mask), 
        1, 
        torch.arange(mask.shape[0], device=embeddings.device).view(-1, 1), 
        0
    )
    mask = mask * logits_mask
    
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    
    return -mean_log_prob_pos.mean()