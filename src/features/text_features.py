"""
Text feature extraction using BERT.

Extracts 768-dimensional [CLS] token embeddings from participant
interview transcripts. Only participant utterances are used to prevent
interviewer bias (Ellie's prompts are excluded).

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department: Computer Science and Engineering, Uttara University
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


class TextFeatureExtractor:
    """Extract BERT embeddings from interview transcripts."""

    def __init__(self, model_name: str = "bert-base-uncased",
                 max_length: int = 512, device: Optional[str] = None):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_transcript(pid, data_dir: str) -> Optional[pd.DataFrame]:
        files = [f for f in os.listdir(data_dir)
                 if f.startswith(str(pid)) and "TRANSCRIPT" in f.upper()]
        if not files:
            return None
        path = os.path.join(data_dir, files[0])
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return None

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _extract_participant_text(self, df: pd.DataFrame) -> str:
        text_col = self._find_column(df, ["value", "text", "Text", "transcript"])
        speaker_col = self._find_column(df, ["speaker", "Speaker", "role"])

        if text_col is None:
            return ""

        if speaker_col is not None:
            mask = df[speaker_col].astype(str).str.contains(
                "Participant", case=False, na=False)
            texts = df[mask][text_col].tolist()
        else:
            texts = df[text_col].tolist()

        return " ".join(str(t) for t in texts if pd.notna(t))

    def extract(self, pid, data_dir: str) -> np.ndarray:
        """Extract 768-dim embedding for a single participant."""
        df = self._load_transcript(pid, data_dir)
        if df is None:
            return np.zeros(768)

        text = self._extract_participant_text(df)
        if len(text.strip()) < 10:
            return np.zeros(768)

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=self.max_length, padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            torch.cuda.empty_cache()
            return embedding
        except Exception:
            return np.zeros(768)
