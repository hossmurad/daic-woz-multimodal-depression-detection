#!/usr/bin/env python3
"""
Feature extraction script for DAIC-WOZ participants.

Extracts text, audio, and visual features for a specified batch of
participants and saves them as .npy files.

Usage:
    python scripts/extract_features.py --participants 300 301 302 --output data/features/batch1

Authors: Md. Murad Hossain, MD. Foysal Ahammed Joy, Nuruzzaman Shuvo
Supervisor: Md. Morshed Ali
Department of Computer Science and Engineering, Uttara University
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.features.text_features import TextFeatureExtractor
from src.features.audio_features import AudioFeatureExtractor
from src.features.visual_features import VisualFeatureExtractor


def extract_batch(participant_ids, data_dir: str, output_dir: str):
    """
    Extract features for a batch of participants.

    Args:
        participant_ids: List of participant IDs
        data_dir: Directory containing raw DAIC-WOZ files
        output_dir: Directory to save extracted features
    """
    os.makedirs(output_dir, exist_ok=True)

    text_extractor = TextFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    visual_extractor = VisualFeatureExtractor()

    all_text, all_audio, all_visual, all_ids = [], [], [], []

    print(f"\nExtracting features for {len(participant_ids)} participants")
    print(f"  Data source: {data_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    start = time.time()

    for idx, pid in enumerate(participant_ids, 1):
        print(f"[{idx}/{len(participant_ids)}] Participant {pid}")

        try:
            text = text_extractor.extract(pid, data_dir)
            audio = audio_extractor.extract(pid, data_dir)
            visual = visual_extractor.extract(pid, data_dir)

            all_text.append(text)
            all_audio.append(audio)
            all_visual.append(visual)
            all_ids.append(int(pid))

            print(f"  Text: {text.shape}, Audio: {audio.shape}, Visual: {visual.shape}")

        except Exception as e:
            print(f"  Error: {e}")
            all_text.append(np.zeros(768))
            all_audio.append(np.zeros(54))
            all_visual.append(np.zeros(50))
            all_ids.append(int(pid))

    elapsed = (time.time() - start) / 60
    print(f"\nExtraction completed in {elapsed:.2f} minutes")

    # Combine into fused features
    all_text = np.array(all_text)
    all_audio = np.array(all_audio)
    all_visual = np.array(all_visual)
    combined = np.hstack([all_text, all_audio, all_visual])

    # Save
    output_path = Path(output_dir)
    np.save(output_path / "text_features.npy", all_text)
    np.save(output_path / "audio_features.npy", all_audio)
    np.save(output_path / "visual_features.npy", all_visual)
    np.save(output_path / "combined_features.npy", combined)
    np.save(output_path / "participant_ids.npy", np.array(all_ids))

    print(f"\nSaved to {output_dir}:")
    print(f"  text_features.npy     {all_text.shape}")
    print(f"  audio_features.npy    {all_audio.shape}")
    print(f"  visual_features.npy   {all_visual.shape}")
    print(f"  combined_features.npy {combined.shape}")
    print(f"  participant_ids.npy   {len(all_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract multimodal features from DAIC-WOZ participants"
    )
    parser.add_argument(
        "--participants", nargs="+", required=True,
        help="List of participant IDs (e.g., 300 301 302)"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing raw DAIC-WOZ files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for extracted features"
    )

    args = parser.parse_args()
    extract_batch(args.participants, args.data_dir, args.output)


if __name__ == "__main__":
    main()
