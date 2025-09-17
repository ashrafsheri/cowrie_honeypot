"""
InfoNCE (Contrastive Learning) fine-tuning for LogBERT.
Optional training approach for learning better sequence representations.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from .train_mlm import LogBertForMaskedLM, LogBertTokenizer

logger = logging.getLogger(__name__)


class InfoNCEHead(nn.Module):
    """InfoNCE contrastive learning head for sequence-level representations."""
    
    def __init__(self, hidden_size: int, projection_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to contrastive space."""
        projections = self.projection(embeddings)
        return F.normalize(projections, dim=-1)


class LogBertWithInfoNCE(nn.Module):
    """LogBERT model with InfoNCE contrastive learning head."""
    
    def __init__(self, pretrained_mlm_path: str, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Load pretrained MLM model
        self.logbert = LogBertForMaskedLM.from_pretrained(pretrained_mlm_path)
        
        # Freeze MLM parameters (optional)
        if config.get('freeze_mlm', False):
            for param in self.logbert.parameters():
                param.requires_grad = False
        
        # InfoNCE head
        hidden_size = self.logbert.config.hidden_size
        projection_dim = config.get('projection_dim', 128)
        temperature = config.get('temperature', 0.07)
        
        self.infonce_head = InfoNCEHead(hidden_size, projection_dim, temperature)
        
        # Next template prediction head (auxiliary task)
        self.next_template_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, config.get('vocab_size', 30522))
        )
        
    def forward(self, input_ids, attention_mask=None, next_template_ids=None, **kwargs):
        """Forward pass for InfoNCE training."""
        # Get embeddings from LogBERT
        embeddings = self.logbert.get_embeddings(input_ids, attention_mask)
        
        # InfoNCE projections
        projections = self.infonce_head(embeddings)
        
        # Next template prediction (auxiliary task)
        next_template_logits = self.next_template_head(embeddings)
        
        # Compute InfoNCE loss
        infonce_loss = self.compute_infonce_loss(projections)
        
        # Next template prediction loss
        next_template_loss = 0
        if next_template_ids is not None:
            next_template_loss = F.cross_entropy(
                next_template_logits, next_template_ids, ignore_index=-100
            )
        
        # Combined loss
        total_loss = infonce_loss + 0.1 * next_template_loss
        
        return {
            'loss': total_loss,
            'infonce_loss': infonce_loss,
            'next_template_loss': next_template_loss,
            'projections': projections,
            'embeddings': embeddings
        }
    
    def compute_infonce_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss."""
        batch_size = projections.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.infonce_head.temperature
        
        # Create positive pairs mask (sequences from same session/source)
        # For simplicity, we treat consecutive sequences as positive pairs
        positive_mask = torch.eye(batch_size, device=projections.device, dtype=torch.bool)
        
        # Create shifted positive pairs
        if batch_size > 1:
            positive_mask[:-1, 1:] |= torch.eye(batch_size - 1, device=projections.device, dtype=torch.bool)
            positive_mask[1:, :-1] |= torch.eye(batch_size - 1, device=projections.device, dtype=torch.bool)
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(False)
        
        # Compute InfoNCE loss
        logits = similarity_matrix
        labels = positive_mask.float()
        
        # Use InfoNCE formulation
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Mean of log-probabilities of positive samples
        mean_log_prob_pos = (labels * log_prob).sum(dim=1) / (labels.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """Get sequence embeddings for inference."""
        return self.logbert.get_embeddings(input_ids, attention_mask)


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with next template prediction."""
    
    def __init__(self, sequences: List[Dict[str, Any]], vocab: Dict[str, int], 
                 tokenizer: LogBertTokenizer, config: Dict[str, Any]):
        self.sequences = sequences
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.config = config
        
        # Group sequences by source and session for positive pair mining
        self.session_groups = self._group_by_session()
        
    def _group_by_session(self) -> Dict[str, List[int]]:
        """Group sequence indices by session/source."""
        groups = {}
        for idx, seq in enumerate(tqdm(self.sequences, desc="Indexing contrastive sessions", leave=False)):
            key = f"{seq['source']}_{seq['host_id']}"
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)
        
        return groups
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Base sequence data
        input_ids = torch.tensor(sequence['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(sequence['attention_mask'], dtype=torch.long)
        
        # Next template prediction target
        next_template_id = -100  # Default ignore index
        
        # Find next sequence in same session for next template prediction
        session_key = f"{sequence['source']}_{sequence['host_id']}"
        if session_key in self.session_groups:
            session_indices = self.session_groups[session_key]
            current_pos = session_indices.index(idx)
            
            if current_pos < len(session_indices) - 1:
                next_idx = session_indices[current_pos + 1]
                next_sequence = self.sequences[next_idx]
                
                # Get first non-special token as next template
                next_input_ids = next_sequence['input_ids']
                for token_id in next_input_ids:
                    if token_id not in [self.vocab.get('[PAD]', 0), 
                                       self.vocab.get('[CLS]', 3), 
                                       self.vocab.get('[SEP]', 4)]:
                        next_template_id = token_id
                        break
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'next_template_ids': torch.tensor(next_template_id, dtype=torch.long),
            'source': sequence['source'],
            'host_id': sequence['host_id'],
            'session_key': session_key
        }


class InfoNCETrainer:
    """Trainer for InfoNCE fine-tuning of LogBERT."""
    
    def __init__(self, config: Dict[str, Any], model_config: Dict[str, Any], 
                 pretrained_mlm_path: str):
        self.config = config
        self.model_config = model_config
        self.pretrained_mlm_path = pretrained_mlm_path
        
        # Output directories
        self.output_dir = config.get('output_dir', './models/logbert-infonce')
        self.logging_dir = config.get('logs_dir', './models/logs')
        
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def prepare_data(self, dataset_path: str) -> tuple:
        """Load and prepare contrastive learning data."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load vocabulary
        vocab_path = os.path.join(dataset_path, 'vocab.json')
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        # Load datasets
        train_df = pd.read_parquet(os.path.join(dataset_path, 'train.parquet'))
        val_df = pd.read_parquet(os.path.join(dataset_path, 'val.parquet'))
        
        # Convert to sequence format
        train_sequences = train_df.to_dict('records')
        val_sequences = val_df.to_dict('records')
        
        # Load tokenizer (should be saved with MLM model)
        tokenizer_path = os.path.join(self.pretrained_mlm_path, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            # Load saved tokenizer
            custom_tokens = self.model_config.get('custom_tokens', ['<IP>', '<USER>', '<PORT>', '<CMD>'])
            self.tokenizer = LogBertTokenizer(
                self.model_config.get('base_model', 'distilbert-base-uncased'), 
                custom_tokens, 
                vocab
            )
        else:
            logger.warning("Tokenizer not found in MLM model path, creating new one")
            custom_tokens = self.model_config.get('custom_tokens', ['<IP>', '<USER>', '<PORT>', '<CMD>'])
            self.tokenizer = LogBertTokenizer(
                self.model_config.get('base_model', 'distilbert-base-uncased'), 
                custom_tokens, 
                vocab
            )
        
        # Create contrastive datasets
        train_dataset = ContrastiveDataset(train_sequences, vocab, self.tokenizer, self.config)
        val_dataset = ContrastiveDataset(val_sequences, vocab, self.tokenizer, self.config)
        
        logger.info(f"Prepared contrastive datasets: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_dataset, val_dataset, vocab
    
    def initialize_model(self, vocab_size: int):
        """Initialize the LogBERT + InfoNCE model."""
        logger.info(f"Initializing LogBERT+InfoNCE model from {self.pretrained_mlm_path}")
        
        # Update config with vocab size
        infonce_config = self.model_config.get('infonce', {})
        infonce_config['vocab_size'] = vocab_size
        
        self.model = LogBertWithInfoNCE(self.pretrained_mlm_path, infonce_config)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"InfoNCE model initialized with {total_params:,} total parameters "
                   f"({trainable_params:,} trainable)")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments for InfoNCE fine-tuning."""
        return TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            
            # Shorter training for fine-tuning
            num_train_epochs=self.config.get('num_train_epochs', 2),
            max_steps=self.config.get('max_steps', -1),
            
            # Smaller batch sizes for contrastive learning
            per_device_train_batch_size=self.config.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.config.get('per_device_eval_batch_size', 16),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),
            
            # Lower learning rate for fine-tuning
            learning_rate=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            warmup_steps=self.config.get('warmup_steps', 100),
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=self.config.get('eval_steps', 250),
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=2,
            
            # Logging
            logging_dir=self.logging_dir,
            logging_steps=self.config.get('logging_steps', 50),
            
            # Performance
            fp16=self.config.get('fp16', True),
            dataloader_num_workers=self.config.get('dataloader_num_workers', 4),
            
            # Other
            seed=42,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Reporting
            report_to=self.config.get('report_to', 'none'),
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for InfoNCE evaluation."""
        # For contrastive learning, the main metric is the loss
        # We can add embedding similarity metrics if needed
        return {}
    
    def train(self, dataset_path: str) -> str:
        """Train InfoNCE model."""
        logger.info("Starting InfoNCE fine-tuning")
        
        # Prepare data
        train_dataset, val_dataset, vocab = self.prepare_data(dataset_path)
        
        # Initialize model
        self.initialize_model(len(vocab))
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.get('early_stopping_patience', 2),
            early_stopping_threshold=self.config.get('early_stopping_threshold', 0.001)
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping]
        )
        
        # Train model
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        if self.tokenizer:
            self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        metrics_path = os.path.join(self.output_dir, 'infonce_training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"InfoNCE training completed. Model saved to {self.output_dir}")
        logger.info(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")
        
        return self.output_dir
    
    def evaluate(self, dataset_path: str, split: str = 'test') -> Dict[str, float]:
        """Evaluate InfoNCE model."""
        if self.trainer is None:
            logger.error("Model not trained yet. Call train() first.")
            return {}
        
        # Load test dataset
        test_df = pd.read_parquet(os.path.join(dataset_path, f'{split}.parquet'))
        test_sequences = test_df.to_dict('records')
        
        # Load vocabulary
        vocab_path = os.path.join(dataset_path, 'vocab.json')
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        test_dataset = ContrastiveDataset(test_sequences, vocab, self.tokenizer, self.config)
        
        # Evaluate
        eval_results = self.trainer.evaluate(test_dataset)
        
        logger.info(f"InfoNCE test evaluation results:")
        for metric, value in eval_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f'{split}_infonce_eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results


def train_infonce_model(config_path: str = './configs/config.yaml', 
                       mlm_model_path: str = './models/logbert-mlm'):
    """Main InfoNCE training function."""
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if InfoNCE is enabled
    if not config['model'].get('use_infonce', False):
        logger.info("InfoNCE training is disabled in config. Skipping.")
        return None, {}
    
    # Initialize trainer
    trainer = InfoNCETrainer(
        config['training'],
        config['model'],
        mlm_model_path
    )
    
    # Start training
    model_path = trainer.train(config['data']['dataset_path'])
    
    # Evaluate on test set
    test_results = trainer.evaluate(config['data']['dataset_path'], 'test')
    
    return model_path, test_results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Train LogBERT InfoNCE model')
    parser.add_argument('--config', default='./configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--mlm-model', required=True,
                       help='Path to pretrained MLM model')
    args = parser.parse_args()

    # Load config and override paths
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['dataset_path'] = args.dataset
    
    # Train InfoNCE model
    model_path, test_results = train_infonce_model(args.config, args.mlm_model)
    
    if model_path:
        print(f"InfoNCE training completed!")
        print(f"Model saved to: {model_path}")
        print(f"Test results: {test_results}")
    else:
        print("InfoNCE training skipped (disabled in config)")
