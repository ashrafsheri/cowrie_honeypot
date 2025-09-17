"""
Masked Language Model (MLM) training for LogBERT.
Primary training approach using HuggingFace Transformers.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, fields
import numpy as np

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BertForMaskedLM, DistilBertForMaskedLM
)
import inspect
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

logger = logging.getLogger(__name__)


class LogBertTokenizer:
    """Custom tokenizer for LogBERT with domain-specific tokens."""
    
    def __init__(self, base_model: str, custom_tokens: List[str], vocab: Dict[str, int]):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.custom_tokens = custom_tokens
        self.vocab = vocab
        
        # Add custom tokens
        self.base_tokenizer.add_tokens(custom_tokens)
        
        # Create reverse vocabulary mapping
        self.id_to_template = {v: k for k, v in vocab.items()}
        
    def tokenize_templates(self, templates: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize template sequences."""
        # Convert templates to text for base tokenizer
        template_texts = []
        for template in templates:
            # Replace template placeholders with domain tokens
            text = template.replace('<*>', '<PLACEHOLDER>')
            template_texts.append(text)
        
        # Use base tokenizer
        encoding = self.base_tokenizer(
            template_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=256
        )
        
        return encoding
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer."""
        self.base_tokenizer.save_pretrained(save_directory)
        
        # Save vocabulary mapping
        vocab_path = os.path.join(save_directory, 'template_vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)


class LogBertForMaskedLM(nn.Module):
    """LogBERT model for Masked Language Modeling."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load base model
        if 'distilbert' in config.base_model:
            self.bert = DistilBertForMaskedLM.from_pretrained(config.base_model)
        else:
            self.bert = BertForMaskedLM.from_pretrained(config.base_model)
        
        # Resize embeddings for custom tokens
        if hasattr(config, 'vocab_size_extended'):
            self.bert.resize_token_embeddings(config.vocab_size_extended)
        
        # Add dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass for MLM training."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """Get embeddings for inference."""
        with torch.no_grad():
            if hasattr(self.bert, 'distilbert'):
                # DistilBERT
                outputs = self.bert.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # BERT
                outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                embeddings = (embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            else:
                embeddings = embeddings.mean(dim=1)
            
            return embeddings


@dataclass
class ModelConfig:
    """Configuration for LogBERT model."""
    base_model: str = "distilbert-base-uncased"
    vocab_size_extended: int = 30522
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    mlm_probability: float = 0.15
    custom_tokens: List[str] = field(default_factory=list)


class LogBertTrainer:
    """Trainer for LogBERT MLM model."""
    
    def __init__(self, config: Dict[str, Any], model_config: Dict[str, Any]):
        self.config = config

        model_field_names = {f.name for f in fields(ModelConfig)}
        model_config_filtered = {
            key: value for key, value in model_config.items() if key in model_field_names
        }
        self.model_config = ModelConfig(**model_config_filtered)
        self.supports_eval_strategy = True

        # Training parameters
        self.output_dir = config.get('output_dir', './models/logbert-mlm')
        self.logging_dir = config.get('logs_dir', './models/logs')
        
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def prepare_data(self, dataset_path: str) -> tuple:
        """Load and prepare training data."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load vocabulary
        vocab_path = os.path.join(dataset_path, 'vocab.json')
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        data_files = {
            'train': os.path.join(dataset_path, 'train.parquet'),
            'validation': os.path.join(dataset_path, 'val.parquet')
        }

        datasets = load_dataset('parquet', data_files=data_files)

        def prepare_split(split_name: str):
            split = datasets[split_name]
            keep_columns = {'input_ids', 'attention_mask', 'labels'}
            drop_columns = [c for c in split.column_names if c not in keep_columns]
            if drop_columns:
                split = split.remove_columns(drop_columns)

            max_key = 'max_train_samples' if split_name == 'train' else 'max_eval_samples'
            max_samples = self.config.get(max_key)
            if max_samples and len(split) > max_samples:
                logger.info(f"Limiting {split_name} split to {max_samples} samples (from {len(split)})")
                split = split.shuffle(seed=42).select(range(max_samples))

            return split

        train_sequences = prepare_split('train')
        val_sequences = prepare_split('validation')

        # Initialize tokenizer
        custom_tokens = self.model_config.custom_tokens or [
            '<IP>', '<USER>', '<PORT>', '<CMD>'
        ]
        self.tokenizer = LogBertTokenizer(
            self.model_config.base_model, 
            custom_tokens, 
            vocab
        )

        # Update vocabulary size
        self.model_config.vocab_size_extended = len(self.tokenizer.base_tokenizer)

        # Create datasets
        train_dataset = train_sequences
        val_dataset = val_sequences

        logger.info(f"Prepared datasets: {len(train_dataset)} train, {len(val_dataset)} val")

        return train_dataset, val_dataset, vocab
    
    def initialize_model(self):
        """Initialize the LogBERT model."""
        logger.info(f"Initializing LogBERT model with base model: {self.model_config.base_model}")
        
        self.model = LogBertForMaskedLM(self.model_config)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model initialized with {total_params:,} total parameters "f"({trainable_params:,} trainable)")
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments."""
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        fp16_enabled = self.config.get('fp16', True) and torch.cuda.is_available()

        if mps_available and not torch.cuda.is_available():
            logger.info("Disabling fp16; MPS backend does not support fp16 training")

        dataloader_workers = self.config.get('dataloader_num_workers', 2)
        if not torch.cuda.is_available() and not mps_available:
            dataloader_workers = 0

        # We'll create a base args_dict with most values from config, but force
        # evaluation_strategy and save_strategy to 'steps' to ensure they match
        learning_rate = self.config.get('learning_rate', 2e-5)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)

        weight_decay = self.config.get('weight_decay', 0.01)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)

        warmup_steps = self.config.get('warmup_steps', 500)
        if isinstance(warmup_steps, str):
            warmup_steps = int(float(warmup_steps))

        eval_strategy_cfg = self.config.get('evaluation_strategy', 'steps')
        save_strategy_cfg = self.config.get('save_strategy', 'steps')

        args_dict = {
            'output_dir': self.output_dir,
            'overwrite_output_dir': True,
            'num_train_epochs': self.config.get('num_train_epochs', 4),
            'max_steps': self.config.get('max_steps', -1),
            'per_device_train_batch_size': self.config.get('per_device_train_batch_size', 16),
            'per_device_eval_batch_size': self.config.get('per_device_eval_batch_size', 32),
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 4),
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'eval_strategy': eval_strategy_cfg,
            'eval_steps': self.config.get('eval_steps', 500),
            'save_strategy': save_strategy_cfg,
            'save_steps': self.config.get('save_steps', 500),
            'save_total_limit': self.config.get('save_total_limit', 3),
            'logging_dir': self.logging_dir,
            'logging_steps': self.config.get('logging_steps', 50),
            'fp16': fp16_enabled,
            'dataloader_num_workers': dataloader_workers,
            'seed': 42,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'report_to': self.config.get('report_to', 'none'),
        }

        # Make sure evaluation_strategy is explicitly set
        # The code below is redundant since we've hardcoded the values above,
        # but keeping it for safety in case someone changes the hardcoded values
        if args_dict.get('load_best_model_at_end', False):
            # Ensure evaluation and save strategy match
            eval_strategy = args_dict.get('eval_strategy', 'steps') or 'steps'
            args_dict['eval_strategy'] = eval_strategy
            args_dict['save_strategy'] = eval_strategy
            if eval_strategy != 'steps':
                logger.info("Setting save_strategy to match eval_strategy=%s for load_best_model_at_end", eval_strategy)

        signature = inspect.signature(TrainingArguments.__init__)
        valid_params = set(signature.parameters.keys()) - {'self'}


        filtered_args = {k: v for k, v in args_dict.items() if k in valid_params}

        return TrainingArguments(**filtered_args)
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        
        # Get predicted token IDs
        predictions = np.argmax(predictions, axis=-1)
        
        # Flatten for metric computation
        predictions_flat = predictions.flatten()
        labels_flat = labels.flatten()
        
        # Ignore padded tokens (-100 is used by HuggingFace for ignored tokens)
        mask = labels_flat != -100
        predictions_masked = predictions_flat[mask]
        labels_masked = labels_flat[mask]
        
        # Compute accuracy
        accuracy = accuracy_score(labels_masked, predictions_masked)
        
        return {
            'accuracy': accuracy,
            'perplexity': np.exp(eval_pred.metrics.get('eval_loss', 0))
        }
    
    def train(self, dataset_path: str) -> str:
        """Train the LogBERT model."""
        logger.info("Starting LogBERT MLM training")
        
        # Prepare data
        train_dataset, val_dataset, vocab = self.prepare_data(dataset_path)
        
        # Initialize model
        self.initialize_model()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator for MLM
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call prepare_data() first.")
            
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer.base_tokenizer,
            mlm=True,
            mlm_probability=self.model_config.mlm_probability
        )
        
        callbacks = []
        if self.supports_eval_strategy and training_args.metric_for_best_model:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.get('early_stopping_patience', 3),
                early_stopping_threshold=self.config.get('early_stopping_threshold', 0.001)
            )
            callbacks.append(early_stopping)
        elif not self.supports_eval_strategy:
            logger.info("Skipping EarlyStoppingCallback because evaluation strategy is not supported by this transformers version.")
        else:
            logger.info("Skipping EarlyStoppingCallback because metric_for_best_model is not defined.")

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # Train model
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metrics
        metrics_path = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        logger.info(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A')}")
        
        return self.output_dir
    
    def evaluate(self, dataset_path: str, split: str = 'test') -> Dict[str, float]:
        """Evaluate trained model on test set."""
        if self.trainer is None:
            logger.error("Model not trained yet. Call train() first.")
            return {}
        
        data_files = {split: os.path.join(dataset_path, f'{split}.parquet')}
        datasets = load_dataset('parquet', data_files=data_files)

        test_dataset = datasets[split]
        keep_columns = {'input_ids', 'attention_mask', 'labels'}
        drop_columns = [c for c in test_dataset.column_names if c not in keep_columns]
        if drop_columns:
            test_dataset = test_dataset.remove_columns(drop_columns)

        max_eval_samples = self.config.get('max_eval_samples') 
        if max_eval_samples and len(test_dataset) > max_eval_samples:
            logger.info(f"Limiting {split} split to {max_eval_samples} samples (from {len(test_dataset)})")
            test_dataset = test_dataset.shuffle(seed=42).select(range(max_eval_samples))

        # Evaluate
        eval_results = self.trainer.evaluate(test_dataset)
        
        logger.info(f"Test evaluation results:")
        for metric, value in eval_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save evaluation results
        eval_path = os.path.join(self.output_dir, f'{split}_eval_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results


def train_mlm_model(config_path: str = './configs/config.yaml'):
    """Main training function."""
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = LogBertTrainer(
        config['training'],
        config['model']
    )
    
    # Start training
    model_path = trainer.train(config['data']['dataset_path'])
    
    # Evaluate on test set
    test_results = trainer.evaluate(config['data']['dataset_path'], 'test')
    
    return model_path, test_results


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Train LogBERT MLM model')
    parser.add_argument('--config', default='./configs/config.yaml',help='Path to configuration file')
    parser.add_argument('--dataset', required=True,help='Path to processed dataset directory') 
    parser.add_argument('--wandb-project', default='logbert-cowrie',help='Weights & Biases project name')
    
    args = parser.parse_args()
    
    # Initialize wandb if available
    if wandb is not None:
        try:
            wandb.init(project=args.wandb_project, name='logbert-mlm-training')
        except Exception:
            logger.warning("wandb not available or not configured")
    else:
        logger.info("wandb not installed; proceeding without experiment tracking")
    
    # Load config and override dataset path
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['dataset_path'] = args.dataset
    
    # Train model
    model_path, test_results = train_mlm_model(args.config)
    
    print(f"Training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Test results: {test_results}")
    
    if wandb is not None and getattr(wandb, 'run', None):
        wandb.finish()
