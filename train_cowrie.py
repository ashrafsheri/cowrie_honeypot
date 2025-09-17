#!/usr/bin/env python
"""
Cowrie honeypot template extraction and training pipeline.
Runs the complete process from log parsing to model training.
"""

import argparse
import logging
import yaml
import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Import project modules
from normalization.cowrie_preprocessor import CowriePreprocessor
from normalization.cowrie_drain import CowrieDrainMiner
from dataset.build_windows import WindowBuilder


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_log_files(config: Dict[str, Any]) -> List[str]:
    """Find all log files according to configuration."""
    files_config = config.get('data', {}).get('files', {})
    log_path = files_config.get('logs_path', './data/logs')
    file_pattern = files_config.get('file_pattern', '*.jsonl')
    
    # Construct glob pattern
    pattern = os.path.join(log_path, file_pattern)
    
    # Find all matching files
    log_files = sorted(glob.glob(pattern))
    
    if not log_files:
        logging.warning(f"No log files found matching pattern: {pattern}")
        
    return log_files


def extract_templates(config: Dict[str, Any], log_files: List[str]) -> pd.DataFrame:
    """Extract templates from log files using Cowrie drain miner."""
    logging.info(f"Extracting templates from {len(log_files)} log files")
    
    # Initialize template miner
    drain_miner = CowrieDrainMiner(config)
    
    # Process log files
    templates_df = drain_miner.process_log_files(log_files)
    
    # Save results
    output_path = config.get('data', {}).get('normalized_path', './data/normalized')
    drain_miner.save_results(output_path, templates_df)
    
    return templates_df


def build_dataset(config: Dict[str, Any], templates_df: pd.DataFrame) -> None:
    """Build dataset for training from extracted templates."""
    logging.info("Building training dataset from templates")
    
    # Initialize window builder
    window_builder = WindowBuilder(config.get('dataset', {}))
    
    # Build sequences
    sequences = window_builder.build_sequences(templates_df)
    
    # Split into train/val/test sets
    train_sequences, val_sequences, test_sequences = window_builder.create_temporal_splits(sequences)
    
    # Save datasets
    output_path = config.get('data', {}).get('dataset_path', './data/dataset')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Build vocabulary from training data
    vocab = window_builder.build_vocabulary(train_sequences)
    
    # Save vocabulary
    vocab_path = os.path.join(output_path, 'vocab.json')
    with open(vocab_path, 'w') as f:
        import json
        json.dump(vocab, f, indent=2)
    
    # Encode sequences
    train_encoded = window_builder.encode_sequences(train_sequences, vocab)
    val_encoded = window_builder.encode_sequences(val_sequences, vocab)
    test_encoded = window_builder.encode_sequences(test_sequences, vocab)
    
    # Save to parquet files
    pd.DataFrame(train_encoded).to_parquet(os.path.join(output_path, "train.parquet"), index=False)
    pd.DataFrame(val_encoded).to_parquet(os.path.join(output_path, "val.parquet"), index=False)
    pd.DataFrame(test_encoded).to_parquet(os.path.join(output_path, "test.parquet"), index=False)
    
    logging.info(f"Saved datasets: train={len(train_sequences)}, val={len(val_sequences)}, test={len(test_sequences)}")


def train_model(config: Dict[str, Any]) -> None:
    """Train the LogBERT model on the prepared dataset."""
    logging.info("Starting model training")
    
    # Import here to avoid circular imports
    from models.train_mlm import train_mlm_model, LogBertTrainer
    
    # Setup trainer directly
    trainer = LogBertTrainer(
        config['training'],
        config['model']
    )
    
    # Start training
    dataset_path = config['data']['dataset_path']
    model_path = trainer.train(dataset_path)
    
    # Evaluate on test set
    test_results = trainer.evaluate(dataset_path, 'test')
    
    logging.info(f"Model training completed. Model saved to {model_path}")
    logging.info(f"Test results: {test_results}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Cowrie Honeypot LogBERT Training Pipeline")
    parser.add_argument("--config", default="./configs/config.yaml", help="Path to config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--skip-templates", action="store_true", help="Skip template extraction")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset building")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Find log files
    log_files = find_log_files(config)
    
    if not log_files:
        logging.error("No log files found. Exiting.")
        return
    
    logging.info(f"Found {len(log_files)} log files to process")
    
    # Extract templates
    templates_df = None
    if not args.skip_templates:
        templates_df = extract_templates(config, log_files)
    else:
        # Load existing templates if skipping extraction
        norm_path = config.get('data', {}).get('normalized_path', './data/normalized')
        template_file = os.path.join(norm_path, "templates.parquet")
        if os.path.exists(template_file):
            templates_df = pd.read_parquet(template_file)
            logging.info(f"Loaded {len(templates_df)} templates from {template_file}")
        else:
            logging.error(f"Template file {template_file} not found. Cannot proceed.")
            return
    
    # Build dataset
    if not args.skip_dataset:
        build_dataset(config, templates_df)
    
    # Train model
    if not args.skip_training:
        train_model(config)
    
    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()