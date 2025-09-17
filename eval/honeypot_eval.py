"""
Cowrie honeypot attack pattern detection evaluation.
Measures effectiveness against common SSH/Telnet attack patterns.
"""

import json
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


class CowrieEvaluator:
    """Evaluates LogBERT model against SSH honeypot attack patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config.get('evaluation', {})
        
        # SSH attack categories
        self.attack_categories = {
            "brute_force": [
                "Multiple failed logins", 
                "Password guessing",
                "Dictionary attacks"
            ],
            "reconnaissance": [
                "System enumeration",
                "Network scanning",
                "User enumeration"
            ],
            "privilege_escalation": [
                "sudo attempts",
                "Exploits", 
                "File permission changes"
            ],
            "malware": [
                "File downloads",
                "Execution attempts",
                "Persistence mechanism"
            ],
            "lateral_movement": [
                "Network connection attempts",
                "Port forwarding",
                "Tunneling"
            ]
        }
        
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset for evaluation."""
        dataset_path = os.path.join(
            self.config.get('data', {}).get('dataset_path', './data/dataset'),
            "test.parquet"
        )
        
        if not os.path.exists(dataset_path):
            logger.error(f"Test dataset not found at {dataset_path}")
            return pd.DataFrame()
        
        return pd.read_parquet(dataset_path)
    
    def inject_attack_patterns(self, test_df: pd.DataFrame, ratio: float = 0.2) -> Tuple[pd.DataFrame, np.ndarray]:
        """Inject synthetic attack patterns into test data."""
        logger.info(f"Injecting attack patterns into {len(test_df)} test sequences")
        
        # Make a copy to avoid modifying original
        df = test_df.copy()
        
        # Determine number of sequences to inject attacks
        n_attacks = int(len(df) * ratio)
        attack_indices = np.random.choice(len(df), n_attacks, replace=False)
        
        # Ground truth labels (0=normal, 1=attack)
        labels = np.zeros(len(df))
        labels[attack_indices] = 1
        
        # Inject attack patterns
        for i, attack_idx in enumerate(attack_indices):
            # Select random attack category
            category = np.random.choice(list(self.attack_categories.keys()))
            patterns = self.attack_categories[category]
            
            # Get original sequence
            seq = df.iloc[attack_idx]
            
            # Inject attack pattern
            if 'templates' in seq and isinstance(seq['templates'], list):
                templates = seq['templates']
                
                # Choose injection point (avoid first and last)
                if len(templates) > 5:
                    inject_idx = np.random.randint(1, len(templates) - 2)
                    
                    # Choose 2-3 attack patterns to inject
                    n_patterns = np.random.randint(2, 4)
                    attack_patterns = [f"<ATTACK:{category}:{pattern}>" 
                                      for pattern in np.random.choice(patterns, n_patterns)]
                    
                    # Insert attack patterns
                    for j, pattern in enumerate(attack_patterns):
                        templates.insert(inject_idx + j, pattern)
                    
                    # Update sequence
                    df.at[attack_idx, 'templates'] = templates
                    df.at[attack_idx, 'attack_category'] = category
                    df.at[attack_idx, 'is_attack'] = True
        
        logger.info(f"Injected {n_attacks} attack sequences ({ratio*100:.1f}% of data)")
        return df, labels
    
    def compute_metrics(self, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics for attack detection."""
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        
        # Compute average precision
        ap = average_precision_score(labels, scores)
        
        # Compute ROC AUC
        auc = roc_auc_score(labels, scores)
        
        # Find best F1 score threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # Compute metrics at different false positive rates
        fpr_thresholds = [0.01, 0.05, 0.1]
        fpr_metrics = {}
        
        # Convert to detection rate at X% FPR
        # (This requires full ROC curve calculation which we skipped for simplicity)
        
        # Return metrics
        return {
            "average_precision": ap,
            "auc": auc,
            "best_threshold": best_threshold,
            "best_f1": best_f1,
            "best_precision": best_precision,
            "best_recall": best_recall,
            "precision_at_90recall": precision[np.argmin(np.abs(recall - 0.9))],
            "threshold_at_90recall": thresholds[np.argmin(np.abs(recall - 0.9))] 
                                    if np.argmin(np.abs(recall - 0.9)) < len(thresholds) else 0.5
        }
    
    def evaluate_by_attack_category(self, test_df: pd.DataFrame, scores: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate performance by attack category."""
        category_metrics = {}
        
        for category in self.attack_categories:
            # Filter to just this category
            category_mask = test_df.get('attack_category', '') == category
            
            if category_mask.sum() == 0:
                logger.warning(f"No samples found for attack category: {category}")
                continue
                
            # Create binary labels (this category vs all others)
            category_labels = category_mask.astype(int).values
            
            # Compute metrics
            category_metrics[category] = self.compute_metrics(category_labels, scores)
            
        return category_metrics
    
    def evaluate_attack_detection(self, model_path: str) -> Dict[str, Any]:
        """Evaluate attack detection performance of trained model."""
        # Load test data
        test_df = self.load_test_data()
        
        if test_df.empty:
            logger.error("No test data available for evaluation")
            return {}
        
        # Inject attack patterns
        test_df_injected, labels = self.inject_attack_patterns(
            test_df, 
            ratio=self.eval_config.get('attack_injection_ratio', 0.2)
        )
        
        # Load model and compute anomaly scores
        # Import here to avoid circular imports
        from inference_service.app import load_model, compute_scores
        
        logger.info(f"Loading model from {model_path}")
        model, tokenizer = load_model(model_path)
        
        logger.info("Computing anomaly scores")
        scores = compute_scores(model, tokenizer, test_df_injected)
        
        # Compute overall metrics
        metrics = self.compute_metrics(labels, scores)
        
        # Compute metrics by attack category
        category_metrics = self.evaluate_by_attack_category(test_df_injected, scores)
        
        # Save evaluation results
        results = {
            "overall": metrics,
            "by_category": category_metrics,
            "model_path": model_path,
            "num_sequences": len(test_df),
            "num_attack_sequences": int(labels.sum()),
            "attack_injection_ratio": self.eval_config.get('attack_injection_ratio', 0.2)
        }
        
        # Save results to file
        output_dir = self.config.get('data', {}).get('evaluation', './data/evaluation')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_file = os.path.join(output_dir, "honeypot_evaluation_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved evaluation results to {output_file}")
        
        return results


def evaluate_honeypot_model(config: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Run honeypot-specific evaluation of trained model."""
    evaluator = CowrieEvaluator(config)
    return evaluator.evaluate_attack_detection(model_path)