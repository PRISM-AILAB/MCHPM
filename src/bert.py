import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
from typing import List


class BertExtractor:

    def __init__(
        self,
        model_ckpt: str = "bert-base-uncased",
        batch_size: int = 8,
        max_length: int = 512,
        use_gpu: bool = True,
    ):
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).to(self.device).eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        embs = []
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
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
            embs.append(cls_emb.cpu())

            del outputs, cls_emb, inputs
            torch.cuda.empty_cache()

        return torch.cat(embs, dim=0)

    def run(self, df: pd.DataFrame, text_col: str, output_col: str = "bert_emb") -> pd.DataFrame:

        if text_col not in df.columns:
            raise KeyError(f"{text_col} column not found in DataFrame.")

        texts = df[text_col].fillna("").astype(str).tolist()
        all_embs = self.encode_texts(texts)

        df = df.copy()
        df[output_col] = all_embs.numpy().tolist()
        return df
