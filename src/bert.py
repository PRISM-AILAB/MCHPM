import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
from typing import List

class BertExtractor:
    """
    Extracts semantic features (768-dim) from text using pre-trained BERT.
    Uses the [CLS] token embedding as the representation.
    """

    def __init__(
        self,
        model_ckpt: str = "bert-base-uncased",
        batch_size: int = 32,
        max_length: int = 128,
        use_gpu: bool = True,
    ):
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"[BertExtractor] Loading model: {model_ckpt} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(self.device).eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encodes a list of strings into a tensor of shape (N, 768).
        """
        embs = []
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT Embedding"):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**inputs)
            # Use the [CLS] token (first token) as the sentence embedding
            cls_emb = outputs.last_hidden_state[:, 0, :]
            embs.append(cls_emb.cpu())

            # Clear cache
            del outputs, cls_emb, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not embs:
            return torch.empty(0, 768)
            
        return torch.cat(embs, dim=0)

    def run(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Helper to run encoding on a DataFrame column and return results.
        NOTE: This is rarely used directly if called from data.py logic,
        but kept for standalone usage.
        """
        if text_col not in df.columns:
            raise KeyError(f"{text_col} not found in DataFrame.")

        texts = df[text_col].astype(str).tolist()
        emb_tensor = self.encode_texts(texts)
        
        # Convert tensor rows to list of numpy arrays for storage in DF
        df['bert_emb'] = list(emb_tensor.numpy())
        return df