import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import copy

# ============================================================
# 1) PyTorch Dataset & DataLoader
# ============================================================

class MultimodalDataset(Dataset):
    def __init__(self, df):
        """
        Expects df columns: 'text_central', 'img_central', 'text_peripheral', 'image_peripheral', 'label'
        """
        # 1. Convert columns to numpy arrays and then tensors
        # stack is important to turn array of arrays into a 2D matrix
        self.bert = torch.tensor(np.stack(df['text_central'].values), dtype=torch.float32)
        self.vgg = torch.tensor(np.stack(df['img_central'].values), dtype=torch.float32)
        
        self.text_per = torch.tensor(np.stack(df['text_peripheral'].values), dtype=torch.float32)
        self.img_per = torch.tensor(np.stack(df['image_peripheral'].values), dtype=torch.float32)
        
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a dictionary of inputs and the label
        inputs = {
            'bert_input': self.bert[idx],
            'vgg16_input': self.vgg[idx],
            'text_peripheral_input': self.text_per[idx],
            'image_peripheral_input': self.img_per[idx]
        }
        return inputs, self.labels[idx]

def get_data_loader(args: dict, df: pd.DataFrame, shuffle: bool = True):
    dataset = MultimodalDataset(df)
    batch_size = int(args.get("batch_size", 128))
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader


# ============================================================
# 2) Model Components (Blocks)
# ============================================================

class CoAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dff, dropout=0.1):
        super(CoAttentionBlock, self).__init__()
        
        # MultiHead Attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Point-wise Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, embed_dim)
        )

    def forward(self, query, key_value):
        query_seq = query.unsqueeze(1)
        key_value_seq = key_value.unsqueeze(1)
        attn_output, _ = self.mha(query_seq, key_value_seq, key_value_seq)
        out1 = self.ln1(query_seq + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.ln2(out1 + self.dropout(ffn_output))
        
        return out2.squeeze(1)


# ============================================================
# 3) Main Model (MCHPM)
# ============================================================

class MCHPM(nn.Module):
    def __init__(self, 
                 feature_dimension=256, 
                 num_heads=4, 
                 dropout=0.1, 
                 dff=2048):
        super(MCHPM, self).__init__()
        
        # --- 1. Local / Global Projections ---
        # BERT path
        self.bert_ln = nn.LayerNorm(768)
        self.bert_proj = nn.Sequential(
            nn.Linear(768, feature_dimension),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # VGG path
        self.vgg_ln = nn.LayerNorm(4096)
        self.vgg_proj = nn.Sequential(
            nn.Linear(4096, feature_dimension),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text Peripheral Path
        self.text_per_net = nn.Sequential(
            nn.Linear(4, 4), nn.ELU(),
            nn.Linear(4, 16), nn.ELU(),
            nn.Linear(16, 64), nn.ELU(),
            nn.Linear(64, 256), nn.ELU(),
            nn.Linear(256, feature_dimension), nn.ELU(),
            nn.LayerNorm(feature_dimension)
        )
        
        # Image Peripheral Path
        self.img_per_net = nn.Sequential(
            nn.Linear(4, 4), nn.ELU(),
            nn.Linear(4, 16), nn.ELU(),
            nn.Linear(16, 64), nn.ELU(),
            nn.Linear(64, 256), nn.ELU(),
            nn.Linear(256, feature_dimension), nn.ELU(),
            nn.LayerNorm(feature_dimension)
        )
        
        # --- 2. Co-Attention Layers ---
        self.co_att_text = CoAttentionBlock(feature_dimension, num_heads, dff, dropout)
        self.co_att_image = CoAttentionBlock(feature_dimension, num_heads, dff, dropout)
        
        # --- 3. Gating Mechanism ---
        self.text_tanh = nn.Sequential(nn.Linear(feature_dimension, feature_dimension), nn.Tanh())
        self.image_tanh = nn.Sequential(nn.Linear(feature_dimension, feature_dimension), nn.Tanh())
        
        self.gate_layer = nn.Sequential(
            nn.Linear(feature_dimension * 2, feature_dimension),
            nn.Sigmoid()
        )
        
        # --- 4. Prediction Head ---
        self.regressor = nn.Sequential(
            nn.Linear(feature_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output 1 score
        )

    def forward(self, inputs):
        bert = inputs['bert_input']
        vgg = inputs['vgg16_input']
        text_per = inputs['text_peripheral_input']
        img_per = inputs['image_peripheral_input']
        
        # 1. Embeddings & Projection
        bert_feat = self.bert_proj(self.bert_ln(bert))
        vgg_feat = self.vgg_proj(self.vgg_ln(vgg))
        
        t_per_feat = self.text_per_net(text_per)
        i_per_feat = self.img_per_net(img_per)
        
        # 2. Combine Main + Peripheral (Feature Fusion Phase 1)
        text_combined = bert_feat + t_per_feat
        img_combined = vgg_feat + i_per_feat
        
        # 3. Co-Attention
        feat_text_from_img = self.co_att_text(query=text_combined, key_value=img_combined)
        feat_img_from_text = self.co_att_image(query=img_combined, key_value=text_combined)
        
        # 4. Gating (Feature Fusion Phase 2)
        h_text = self.text_tanh(feat_text_from_img)
        h_image = self.image_tanh(feat_img_from_text)
        concat_feat = torch.cat([feat_text_from_img, feat_img_from_text], dim=1)
        z = self.gate_layer(concat_feat) 
        
        gated_text = torch.mul(z, h_text)
        gated_image = torch.mul(1.0 - z, h_image)
        
        fused = gated_text + gated_image
        
        # 5. Prediction
        output = self.regressor(fused)
        return output

def RHP(feature_dimension, num_heads, dropout=0.1, learning_rate=1e-4, **kwargs):
    model = MCHPM(feature_dimension, num_heads, dropout)
    return model


# ============================================================
# 4) Trainer & Tester
# ============================================================

def rhp_trainer(args, model, train_loader, val_loader, best_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Using device: {device}")
    
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=float(args.get("lr", 1e-4)))
    criterion = nn.MSELoss()
    
    epochs = int(args.get("num_epochs", 100))
    patience = int(args.get("patience", 10))
    
    best_val_loss = float('inf')
    counter = 0 # Early stopping counter
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device).unsqueeze(1) # [Batch, 1]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- Early Stopping & Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> Model saved! Improved val_loss: {best_val_loss:.4f}")
        else:
            counter += 1
            print(f"   >>> No improvement. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("   >>> Early Stopping Triggered.")
                break
                
    return None


def rhp_tester(args, model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    preds_list = []
    trues_list = []
    
    print("[Tester] Starting Inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            
            outputs = model(inputs)
            
            # Move back to CPU for numpy conversion
            preds_list.append(outputs.cpu().numpy())
            trues_list.append(labels.numpy())
            
    preds = np.concatenate(preds_list).flatten()
    trues = np.concatenate(trues_list).flatten()
    
    return preds, trues