"""
Dataset building module for LogBERT training.
Handles time-based windowing, sequence building, and train/val/test splits.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

logger = logging.getLogger(__name__)


class WindowBuilder:
    """Builds time-based windows from normalized log sequences."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_length = config.get('window_length', 12)
        self.stride = config.get('stride', 1)
        self.min_window_length = config.get('min_window_length', 8)
        self.max_sequence_length = config.get('max_sequence_length', 256)
        
        # Split ratios
        self.train_split = config.get('train_split', 0.70)
        self.val_split = config.get('val_split', 0.15)
        self.test_split = config.get('test_split', 0.15)
        
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
    
    def build_sequences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build sequences from normalized log data grouped by (source, host_id)."""
        sequences = []
        
        # Group by source and host_id for temporal ordering
        grouped = df.groupby(['source', 'host_id'])
        
        for (source, host_id), group in grouped:
            # Sort by timestamp to ensure temporal order
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            if len(group) < self.min_window_length:
                logger.debug(f"Skipping group ({source}, {host_id}) with only {len(group)} entries")
                continue
            
            # Extract templates as the primary sequence
            templates = group['template'].tolist()
            
            # Build sliding windows
            group_sequences = self._build_sliding_windows(
                templates, group, source, host_id
            )
            
            sequences.extend(group_sequences)
        
        logger.info(f"Built {len(sequences)} sequences from {len(df)} log entries")
        return sequences
    
    def _build_sliding_windows(self, templates: List[str], group: pd.DataFrame, 
                              source: str, host_id: str) -> List[Dict[str, Any]]:
        """Build sliding windows from a single source/host sequence."""
        windows = []
        
        for i in range(0, len(templates) - self.min_window_length + 1, self.stride):
            end_idx = min(i + self.window_length, len(templates))
            
            if end_idx - i < self.min_window_length:
                break
                
            # Extract window data
            window_templates = templates[i:end_idx]
            window_group = group.iloc[i:end_idx]
            
            # Build sequence metadata
            sequence_data = {
                'source': source,
                'host_id': host_id,
                'start_time': window_group.iloc[0]['timestamp'],
                'end_time': window_group.iloc[-1]['timestamp'],
                'templates': window_templates,
                'template_ids': window_group['template_id'].tolist(),
                'severities': window_group['severity'].tolist(),
                'session_ids': window_group['session_id'].tolist(),
                'parameters': [json.loads(p) if p else {} for p in window_group['parameters']],
                'sequence_length': len(window_templates),
                'duration_seconds': (window_group.iloc[-1]['timestamp'] - 
                                   window_group.iloc[0]['timestamp']).total_seconds()
            }
            
            windows.append(sequence_data)
        
        return windows
    
    def create_temporal_splits(self, sequences: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        """Create train/val/test splits based on temporal ordering."""
        # Sort sequences by start time
        sorted_sequences = sorted(sequences, key=lambda x: x['start_time'])
        
        n_total = len(sorted_sequences)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        # Time-based splits to avoid data leakage
        train_sequences = sorted_sequences[:n_train]
        val_sequences = sorted_sequences[n_train:n_train + n_val]
        test_sequences = sorted_sequences[n_train + n_val:]
        
        logger.info(f"Created temporal splits: {len(train_sequences)} train, "
                   f"{len(val_sequences)} val, {len(test_sequences)} test")
        
        # Log time ranges for verification
        def log_time_range(seqs, name):
            if seqs:
                start = min(s['start_time'] for s in seqs)
                end = max(s['end_time'] for s in seqs)
                logger.info(f"{name} time range: {start} to {end}")
        
        log_time_range(train_sequences, "Train")
        log_time_range(val_sequences, "Val")
        log_time_range(test_sequences, "Test")
        
        return train_sequences, val_sequences, test_sequences
    
    def build_vocabulary(self, sequences: List[Dict[str, Any]]) -> Dict[str, int]:
        """Build template vocabulary from training sequences."""
        template_counts = {}
        
        for seq in sequences:
            for template in seq['templates']:
                template_counts[template] = template_counts.get(template, 0) + 1
        
        # Sort templates by frequency
        sorted_templates = sorted(template_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Build vocabulary with special tokens
        vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[MASK]': 2,
            '[CLS]': 3,
            '[SEP]': 4
        }
        
        # Add templates to vocabulary
        for template, count in sorted_templates:
            if template not in vocab:
                vocab[template] = len(vocab)
        
        logger.info(f"Built vocabulary with {len(vocab)} tokens "
                   f"({len(sorted_templates)} unique templates)")
        
        return vocab
    
    def encode_sequences(self, sequences: List[Dict[str, Any]], 
                        vocab: Dict[str, int]) -> List[Dict[str, Any]]:
        """Encode template sequences using vocabulary."""
        encoded_sequences = []
        
        for seq in sequences:
            # Encode templates to token IDs
            token_ids = []
            for template in seq['templates']:
                token_id = vocab.get(template, vocab['[UNK]'])
                token_ids.append(token_id)
            
            # Pad or truncate to max sequence length
            if len(token_ids) > self.max_sequence_length - 2:  # Reserve space for [CLS], [SEP]
                token_ids = token_ids[:self.max_sequence_length - 2]
            
            # Add special tokens
            input_ids = [vocab['[CLS]']] + token_ids + [vocab['[SEP]']]
            
            # Pad to max length
            attention_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_sequence_length:
                input_ids.append(vocab['[PAD]'])
                attention_mask.append(0)
            
            # Build encoded sequence
            encoded_seq = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': input_ids.copy(),  # For MLM, labels are same as input_ids
                'source': seq['source'],
                'host_id': seq['host_id'],
                'start_time': seq['start_time'].isoformat(),
                'end_time': seq['end_time'].isoformat(),
                'sequence_length': seq['sequence_length'],
                'duration_seconds': seq['duration_seconds'],
                'severities': seq['severities'],
                'session_ids': seq['session_ids']
            }
            
            encoded_sequences.append(encoded_seq)
        
        return encoded_sequences
    
    def add_synthetic_anomalies(self, sequences: List[Dict[str, Any]], 
                               vocab: Dict[str, int], 
                               anomaly_ratio: float = 0.05) -> List[Dict[str, Any]]:
        """Add synthetic anomalies for evaluation purposes."""
        anomalous_sequences = []
        n_anomalies = int(len(sequences) * anomaly_ratio)
        
        # Select random sequences to make anomalous
        anomaly_indices = np.random.choice(len(sequences), n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            seq = sequences[idx].copy()
            
            # Apply anomaly transformations
            seq = self._apply_anomaly_transformation(seq, vocab)
            seq['is_anomaly'] = True
            
            anomalous_sequences.append(seq)
        
        # Mark normal sequences
        for i, seq in enumerate(sequences):
            if i not in anomaly_indices:
                seq['is_anomaly'] = False
        
        logger.info(f"Added {len(anomalous_sequences)} synthetic anomalies "
                   f"({anomaly_ratio:.1%} of dataset)")
        
        return sequences + anomalous_sequences
    
    def _apply_anomaly_transformation(self, seq: Dict[str, Any], 
                                    vocab: Dict[str, int]) -> Dict[str, Any]:
        """Apply anomaly transformations to a sequence."""
        input_ids = seq['input_ids'].copy()
        
        # Types of anomaly transformations
        transformations = [
            self._inject_rare_tokens,
            self._shuffle_subsequence, 
            self._repeat_tokens,
            self._insert_out_of_order
        ]
        
        # Apply random transformation
        transform = np.random.choice(transformations)
        input_ids = transform(input_ids, vocab)
        
        seq['input_ids'] = input_ids
        seq['labels'] = input_ids.copy()
        
        return seq
    
    def _inject_rare_tokens(self, input_ids: List[int], vocab: Dict[str, int]) -> List[int]:
        """Inject rare/unseen tokens into sequence."""
        # Replace 1-2 random tokens with UNK
        n_replace = np.random.randint(1, 3)
        valid_positions = [i for i, token_id in enumerate(input_ids) 
                          if token_id not in [vocab['[PAD]'], vocab['[CLS]'], vocab['[SEP]']]]
        
        if valid_positions:
            replace_positions = np.random.choice(valid_positions, 
                                               min(n_replace, len(valid_positions)), 
                                               replace=False)
            for pos in replace_positions:
                input_ids[pos] = vocab['[UNK]']
        
        return input_ids
    
    def _shuffle_subsequence(self, input_ids: List[int], vocab: Dict[str, int]) -> List[int]:
        """Shuffle a subsequence to break temporal order."""
        valid_positions = [i for i, token_id in enumerate(input_ids) 
                          if token_id not in [vocab['[PAD]'], vocab['[CLS]'], vocab['[SEP]']]]
        
        if len(valid_positions) >= 3:
            start_idx = np.random.randint(0, len(valid_positions) - 2)
            end_idx = min(start_idx + np.random.randint(2, 5), len(valid_positions))
            
            # Shuffle subsequence
            subseq_positions = valid_positions[start_idx:end_idx]
            subseq_values = [input_ids[i] for i in subseq_positions]
            np.random.shuffle(subseq_values)
            
            for pos, val in zip(subseq_positions, subseq_values):
                input_ids[pos] = val
        
        return input_ids
    
    def _repeat_tokens(self, input_ids: List[int], vocab: Dict[str, int]) -> List[int]:
        """Repeat tokens to simulate stuck/looping behavior."""
        valid_positions = [i for i, token_id in enumerate(input_ids) 
                          if token_id not in [vocab['[PAD]'], vocab['[CLS]'], vocab['[SEP]']]]
        
        if valid_positions:
            # Pick a token to repeat
            repeat_pos = np.random.choice(valid_positions)
            repeat_token = input_ids[repeat_pos]
            
            # Find positions to replace with repeated token
            n_repeats = np.random.randint(1, 4)
            replace_positions = np.random.choice(valid_positions, 
                                               min(n_repeats, len(valid_positions)), 
                                               replace=False)
            
            for pos in replace_positions:
                input_ids[pos] = repeat_token
        
        return input_ids
    
    def _insert_out_of_order(self, input_ids: List[int], vocab: Dict[str, int]) -> List[int]:
        """Insert tokens in wrong chronological position."""
        # This is a simplified version - in practice would use domain knowledge
        return self._shuffle_subsequence(input_ids, vocab)


class DatasetBuilder:
    """Main dataset builder orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_builder = WindowBuilder(config)
        self.output_format = config.get('output_format', 'parquet')
        
    def build_dataset(self, normalized_df: pd.DataFrame, 
                     output_path: str) -> Dict[str, str]:
        """Build complete dataset with train/val/test splits."""
        logger.info(f"Building dataset from {len(normalized_df)} normalized log entries")
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Build sequences
        sequences = self.window_builder.build_sequences(normalized_df)
        
        # Step 2: Create temporal splits
        train_seqs, val_seqs, test_seqs = self.window_builder.create_temporal_splits(sequences)
        
        # Step 3: Build vocabulary from training data
        vocab = self.window_builder.build_vocabulary(train_seqs)
        
        # Save vocabulary
        vocab_path = output_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        # Step 4: Encode sequences
        train_encoded = self.window_builder.encode_sequences(train_seqs, vocab)
        val_encoded = self.window_builder.encode_sequences(val_seqs, vocab)
        test_encoded = self.window_builder.encode_sequences(test_seqs, vocab)
        
        # Step 5: Add synthetic anomalies to test set for evaluation
        test_encoded_with_anomalies = self.window_builder.add_synthetic_anomalies(
            test_encoded, vocab, anomaly_ratio=0.1
        )
        
        # Step 6: Save datasets
        output_paths = {}
        
        datasets = {
            'train': train_encoded,
            'val': val_encoded, 
            'test': test_encoded_with_anomalies
        }
        
        for split_name, data in datasets.items():
            output_file = output_dir / f'{split_name}.{self.output_format}'
            
            if self.output_format == 'parquet':
                df_split = pd.DataFrame(data)
                df_split.to_parquet(output_file, index=False)
            elif self.output_format == 'jsonl':
                with open(output_file, 'w') as f:
                    for item in data:
                        f.write(json.dumps(item) + '\n')
            else:
                raise ValueError(f"Unsupported output format: {self.output_format}")
            
            output_paths[split_name] = str(output_file)
            logger.info(f"Saved {len(data)} {split_name} sequences to {output_file}")
        
        # Save dataset statistics
        stats = {
            'total_sequences': len(sequences),
            'train_sequences': len(train_encoded),
            'val_sequences': len(val_encoded),
            'test_sequences': len(test_encoded_with_anomalies),
            'vocab_size': len(vocab),
            'window_length': self.window_builder.window_length,
            'max_sequence_length': self.window_builder.max_sequence_length,
            'created_at': datetime.now().isoformat()
        }
        
        stats_path = output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        output_paths['vocab'] = str(vocab_path)
        output_paths['stats'] = str(stats_path)
        
        logger.info(f"Dataset building complete. Statistics:")
        logger.info(f"  Total sequences: {stats['total_sequences']}")
        logger.info(f"  Train/Val/Test: {stats['train_sequences']}/{stats['val_sequences']}/{stats['test_sequences']}")
        logger.info(f"  Vocabulary size: {stats['vocab_size']}")
        
        return output_paths


def load_dataset(dataset_path: str, split: str = 'train') -> pd.DataFrame:
    """Load a dataset split."""
    file_path = Path(dataset_path) / f'{split}.parquet'
    
    if not file_path.exists():
        # Try JSONL format
        file_path = Path(dataset_path) / f'{split}.jsonl'
        if file_path.exists():
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return pd.DataFrame(data)
        else:
            raise FileNotFoundError(f"Dataset split not found: {file_path}")
    
    return pd.read_parquet(file_path)


def load_vocabulary(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Example usage for testing
    import yaml
    
    # Load configuration
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load normalized data (assuming it exists)
    try:
        normalized_df = pd.read_parquet('./data/normalized/cowrie_normalized.parquet')
        
        # Build dataset
        builder = DatasetBuilder(config['dataset'])
        output_paths = builder.build_dataset(normalized_df, './data/dataset')
        
        print("Dataset building complete!")
        print("Output files:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")
            
    except FileNotFoundError:
        print("Normalized data not found. Run normalization first.")
        print("Example: python normalization/drain_miner.py")