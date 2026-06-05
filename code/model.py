import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig, MistralModel, MistralConfig
from peft import LoraConfig, get_peft_model

class FlowTokenizer(nn.Module):
    def __init__(self, input_dim=9, embed_dim=4096):
        super(FlowTokenizer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class ClassificationProjection(nn.Module):
    def __init__(self, embed_dim=4096, hidden_dim=256):
        super(ClassificationProjection, self).__init__()
        self.linear_proj = nn.Linear(embed_dim, hidden_dim)
        self.classification_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.linear_proj(x)
        x = torch.relu(x)
        logits = self.classification_layer(x)
        return logits

class DoLLM(nn.Module):
    def __init__(self, model_path="meta-llama/Llama-2-7b-chat-hf"):
        super(DoLLM, self).__init__()
        self.flow_tokenizer = FlowTokenizer()
        
        config = LlamaConfig.from_pretrained(model_path)
        config.is_decoder = False 
        
        self.backbone = LlamaModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.classification_projection = ClassificationProjection()

    def forward(self, flow_sequence):
        token_embeddings = self.flow_tokenizer(flow_sequence).to(torch.bfloat16)
        
        attention_mask = torch.ones(
            token_embeddings.shape[:2], 
            device=token_embeddings.device, 
            dtype=torch.long
        )
        
        outputs = self.backbone(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state.to(torch.float32)
        
        logits = self.classification_projection(last_hidden_state)
        
        return logits

class DoLLMLora(nn.Module):
    def __init__(self, model_path="mistralai/Mistral-7B-v0.1"):
        super(DoLLMLora, self).__init__()
        self.flow_tokenizer = FlowTokenizer()
        
        config = MistralConfig.from_pretrained(model_path)
        config.is_decoder = False 
        
        base_model = MistralModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        self.backbone = get_peft_model(base_model, lora_config)
        self.classification_projection = ClassificationProjection()

    def forward(self, flow_sequence):
        token_embeddings = self.flow_tokenizer(flow_sequence).to(torch.bfloat16)
        
        attention_mask = torch.ones(
            token_embeddings.shape[:2], 
            device=token_embeddings.device, 
            dtype=torch.long
        )
        
        outputs = self.backbone(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state.to(torch.float32)
        logits = self.classification_projection(last_hidden_state)
        
        return logits
    
class DoLLMConv(nn.Module):
    def __init__(self, model_path="mistralai/Mistral-7B-v0.1"):
        super(DoLLMConv, self).__init__()

        self.flow_tokenizer = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.tokenizer_projection = nn.Linear(512, 4096)

        config = MistralConfig.from_pretrained(model_path)
        config.is_decoder = False
        config.output_hidden_states = True

        self.backbone = MistralModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.attention_pool = nn.MultiheadAttention(
            embed_dim=4096, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )

        self.classification_head = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2)
        )

    def forward(self, flow_sequence, return_hidden=False):
        x_conv = flow_sequence.transpose(1, 2)
        
        conv_out = self.flow_tokenizer(x_conv)
        conv_out = conv_out.transpose(1, 2)
        
        token_embeddings = self.tokenizer_projection(conv_out).to(torch.bfloat16)

        attention_mask = torch.ones(
            token_embeddings.shape[:2],
            device=token_embeddings.device,
            dtype=torch.long
        )

        with torch.no_grad():
            outputs = self.backbone(
                inputs_embeds=token_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        last_hidden_state = outputs.hidden_states[-1]

        attn_out, _ = self.attention_pool(
            query=last_hidden_state,
            key=last_hidden_state,
            value=last_hidden_state
        )

        combined_features = torch.cat([attn_out, token_embeddings], dim=-1)
        
        logits = self.classification_head(combined_features)

        if return_hidden:
            return logits, combined_features
        return logits