"""
Comprehensive metrics computation for LogBERT anomaly detection evaluation.
Handles recall on synthetic attacks, precision@K, detection latency, and drift monitoring.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SyntheticAttackEvaluator:
    """Evaluator for synthetic attack campaign recall."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recall_threshold = config.get('recall_threshold', 0.9)
        
    def evaluate_campaign_recall(self, predictions_df: pd.DataFrame, 
                                campaigns_file: str) -> Dict[str, float]:
        """Evaluate recall on synthetic attack campaigns."""
        
        # Load campaign metadata
        with open(campaigns_file, 'r') as f:
            campaigns = json.load(f)
        
        results = {}
        
        for campaign in campaigns:
            campaign_id = campaign['campaign_id']
            attack_type = campaign['attack_type']
            start_time = pd.to_datetime(campaign['start_time'])
            end_time = pd.to_datetime(campaign['end_time'])
            target_sources = campaign['target_sources']
            
            # Find predictions for this campaign
            campaign_mask = (
                (pd.to_datetime(predictions_df['timestamp']) >= start_time) &
                (pd.to_datetime(predictions_df['timestamp']) <= end_time) &
                (predictions_df['source'].isin(target_sources))
            )
            
            campaign_predictions = predictions_df[campaign_mask]
            
            if len(campaign_predictions) == 0:
                logger.warning(f"No predictions found for campaign {campaign_id}")
                results[campaign_id] = 0.0
                continue
            
            # Calculate recall (fraction of campaign sequences detected)
            if 'score_ml' in campaign_predictions.columns:
                # Use threshold to determine detections
                threshold = self.config.get('detection_threshold', 0.5)
                detected = (campaign_predictions['score_ml'] >= threshold).sum()
                total = len(campaign_predictions)
                
                recall = detected / total if total > 0 else 0.0
                results[campaign_id] = recall
                
                logger.info(f"Campaign {campaign_id} ({attack_type}): "
                           f"Recall = {recall:.3f} ({detected}/{total})")
            else:
                logger.error(f"No score_ml column found in predictions")
                results[campaign_id] = 0.0
        
        # Aggregate results
        if results:
            overall_recall = np.mean(list(results.values()))
            attack_type_recalls = {}
            
            for campaign in campaigns:
                attack_type = campaign['attack_type']
                campaign_id = campaign['campaign_id']
                
                if attack_type not in attack_type_recalls:
                    attack_type_recalls[attack_type] = []
                
                if campaign_id in results:
                    attack_type_recalls[attack_type].append(results[campaign_id])
            
            # Average by attack type
            for attack_type, recalls in attack_type_recalls.items():
                attack_type_recalls[attack_type] = np.mean(recalls)
            
            summary = {
                'overall_recall': overall_recall,
                'attack_type_recalls': attack_type_recalls,
                'individual_campaigns': results,
                'meets_threshold': overall_recall >= self.recall_threshold
            }
            
            return summary
        else:
            return {'overall_recall': 0.0, 'meets_threshold': False}


class PrecisionAtKEvaluator:
    """Evaluator for precision@K metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.precision_k = config.get('precision_k', 50)
        self.manual_sample_size = config.get('manual_sample_size', 100)
        
    def evaluate_precision_at_k(self, predictions_df: pd.DataFrame, 
                               weak_labels_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Evaluate precision@K using weak labels or manual sampling."""
        
        if 'score_ml' not in predictions_df.columns:
            logger.error("No score_ml column found in predictions")
            return {'precision_at_k': 0.0, 'error': 'No scores found'}
        
        # Sort by anomaly score (descending)
        sorted_df = predictions_df.sort_values('score_ml', ascending=False)
        
        # Get top-K predictions
        top_k = sorted_df.head(self.precision_k)
        
        if weak_labels_df is not None and 'weak_label' in weak_labels_df.columns:
            # Use weak labels for precision estimation
            
            # Match predictions with weak labels (assuming same index or key)
            if 'sequence_id' in top_k.columns and 'sequence_id' in weak_labels_df.columns:
                merged = top_k.merge(weak_labels_df[['sequence_id', 'weak_label']], 
                                   on='sequence_id', how='left')
            else:
                # Assume same order/index
                merged = top_k.copy()
                merged['weak_label'] = weak_labels_df['weak_label'].iloc[:len(top_k)].values
            
            # Calculate precision
            true_positives = merged['weak_label'].sum()
            precision_at_k = true_positives / len(merged)
            
            # Also calculate precision at different K values
            precision_at_ks = {}
            for k in [10, 20, 50, 100]:
                if k <= len(sorted_df):
                    top_k_temp = sorted_df.head(k)
                    if 'sequence_id' in top_k_temp.columns:
                        merged_temp = top_k_temp.merge(
                            weak_labels_df[['sequence_id', 'weak_label']], 
                            on='sequence_id', how='left'
                        )
                    else:
                        merged_temp = top_k_temp.copy()
                        merged_temp['weak_label'] = weak_labels_df['weak_label'].iloc[:k].values
                    
                    tp = merged_temp['weak_label'].sum()
                    precision_at_ks[f'precision_at_{k}'] = tp / k
            
            return {
                'precision_at_k': precision_at_k,
                'k': self.precision_k,
                'true_positives': int(true_positives),
                **precision_at_ks
            }
        
        else:
            # Manual evaluation placeholder
            logger.warning("No weak labels provided. Manual evaluation required.")
            
            # Save top-K for manual review
            manual_review_path = './data/manual_review_topk.csv'
            top_k.to_csv(manual_review_path, index=False)
            
            return {
                'precision_at_k': 0.0,  # Placeholder
                'k': self.precision_k,
                'manual_review_file': manual_review_path,
                'note': 'Manual evaluation required - check manual_review_file'
            }


class LatencyEvaluator:
    """Evaluator for detection latency metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_latency_p95 = config.get('detection_latency_p95', 10)  # seconds
        
    def evaluate_detection_latency(self, predictions_df: pd.DataFrame, 
                                  campaigns_file: str) -> Dict[str, float]:
        """Evaluate end-to-end detection latency."""
        
        # Load campaign metadata
        with open(campaigns_file, 'r') as f:
            campaigns = json.load(f)
        
        latencies = []
        
        for campaign in campaigns:
            campaign_id = campaign['campaign_id']
            start_time = pd.to_datetime(campaign['start_time'])
            target_sources = campaign['target_sources']
            
            # Find first detection for this campaign
            campaign_mask = (
                (pd.to_datetime(predictions_df['timestamp']) >= start_time) &
                (predictions_df['source'].isin(target_sources))
            )
            
            campaign_predictions = predictions_df[campaign_mask].copy()
            
            if len(campaign_predictions) == 0:
                continue
            
            # Sort by timestamp
            campaign_predictions = campaign_predictions.sort_values('timestamp')
            
            # Find first detection above threshold
            threshold = self.config.get('detection_threshold', 0.5)
            detections = campaign_predictions[campaign_predictions['score_ml'] >= threshold]
            
            if len(detections) > 0:
                first_detection_time = pd.to_datetime(detections.iloc[0]['timestamp'])
                latency = (first_detection_time - start_time).total_seconds()
                latencies.append(latency)
                
                logger.debug(f"Campaign {campaign_id}: Detection latency = {latency:.1f}s")
        
        if latencies:
            return {
                'mean_latency': np.mean(latencies),
                'median_latency': np.median(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'min_latency': np.min(latencies),
                'max_latency': np.max(latencies),
                'total_campaigns_detected': len(latencies),
                'meets_p95_target': np.percentile(latencies, 95) <= self.target_latency_p95
            }
        else:
            return {
                'mean_latency': float('inf'),
                'p95_latency': float('inf'),
                'total_campaigns_detected': 0,
                'meets_p95_target': False
            }


class DriftMonitor:
    """Monitor for distribution drift in scores and templates."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.js_divergence_threshold = config.get('js_divergence_threshold', 0.1)
        self.ks_test_p_value = config.get('ks_test_p_value', 0.05)
        self.baseline_window_days = config.get('baseline_window_days', 30)
        
    def compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Compute JS divergence
        m = (p + q) / 2
        js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
        
        return js_div
    
    def evaluate_template_drift(self, current_df: pd.DataFrame, 
                               baseline_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate drift in template distributions."""
        
        # Get template distributions
        current_templates = current_df['template'].value_counts(normalize=True)
        baseline_templates = baseline_df['template'].value_counts(normalize=True)
        
        # Align distributions (use union of templates)
        all_templates = set(current_templates.index) | set(baseline_templates.index)
        
        current_dist = np.array([current_templates.get(t, 0) for t in all_templates])
        baseline_dist = np.array([baseline_templates.get(t, 0) for t in all_templates])
        
        # Compute JS divergence
        js_div = self.compute_js_divergence(current_dist, baseline_dist)
        
        # Perform KS test on template frequencies
        current_freqs = current_df['template'].value_counts().values
        baseline_freqs = baseline_df['template'].value_counts().values
        
        ks_stat, ks_p_value = stats.ks_2samp(current_freqs, baseline_freqs)
        
        return {
            'template_js_divergence': js_div,
            'template_ks_statistic': ks_stat,
            'template_ks_p_value': ks_p_value,
            'template_drift_detected': js_div > self.js_divergence_threshold,
            'template_distribution_change': ks_p_value < self.ks_test_p_value
        }
    
    def evaluate_score_drift(self, current_scores: np.ndarray, 
                            baseline_scores: np.ndarray) -> Dict[str, float]:
        """Evaluate drift in score distributions."""
        
        # KS test for distribution change
        ks_stat, ks_p_value = stats.ks_2samp(current_scores, baseline_scores)
        
        # Compare distribution moments
        current_mean = np.mean(current_scores)
        baseline_mean = np.mean(baseline_scores)
        mean_shift = abs(current_mean - baseline_mean)
        
        current_std = np.std(current_scores)
        baseline_std = np.std(baseline_scores)
        std_ratio = current_std / baseline_std if baseline_std > 0 else 1.0
        
        # Compute histograms for JS divergence
        bins = np.linspace(
            min(np.min(current_scores), np.min(baseline_scores)),
            max(np.max(current_scores), np.max(baseline_scores)),
            50
        )
        
        current_hist, _ = np.histogram(current_scores, bins=bins, density=True)
        baseline_hist, _ = np.histogram(baseline_scores, bins=bins, density=True)
        
        js_div = self.compute_js_divergence(current_hist, baseline_hist)
        
        return {
            'score_js_divergence': js_div,
            'score_ks_statistic': ks_stat,
            'score_ks_p_value': ks_p_value,
            'score_mean_shift': mean_shift,
            'score_std_ratio': std_ratio,
            'score_drift_detected': ks_p_value < self.ks_test_p_value
        }


class AlertVolumeAnalyzer:
    """Analyze alert volume and patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def analyze_alert_volume(self, predictions_df: pd.DataFrame, 
                           threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze alert volume patterns."""
        
        # Apply threshold to get alerts
        alerts_df = predictions_df[predictions_df['score_ml'] >= threshold].copy()
        
        if len(alerts_df) == 0:
            return {
                'total_alerts': 0,
                'alerts_per_day': 0.0,
                'alert_rate': 0.0
            }
        
        # Convert timestamp and analyze patterns
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Time range analysis
        min_time = alerts_df['timestamp'].min()
        max_time = alerts_df['timestamp'].max()
        time_span_days = (max_time - min_time).total_seconds() / 86400
        
        if time_span_days == 0:
            time_span_days = 1  # Avoid division by zero
        
        # Volume metrics
        total_alerts = len(alerts_df)
        alerts_per_day = total_alerts / time_span_days
        alert_rate = total_alerts / len(predictions_df)
        
        # Source analysis
        top_sources = alerts_df['source'].value_counts().head(10)
        source_diversity = alerts_df['source'].nunique()
        
        # Temporal patterns
        alerts_df['hour'] = alerts_df['timestamp'].dt.hour
        hourly_pattern = alerts_df['hour'].value_counts().sort_index()
        
        # Score distribution
        score_stats = {
            'mean_score': alerts_df['score_ml'].mean(),
            'median_score': alerts_df['score_ml'].median(),
            'score_std': alerts_df['score_ml'].std(),
            'min_score': alerts_df['score_ml'].min(),
            'max_score': alerts_df['score_ml'].max()
        }
        
        return {
            'total_alerts': total_alerts,
            'alerts_per_day': alerts_per_day,
            'alert_rate': alert_rate,
            'time_span_days': time_span_days,
            'source_diversity': source_diversity,
            'top_alert_sources': top_sources.to_dict(),
            'hourly_alert_pattern': hourly_pattern.to_dict(),
            'score_statistics': score_stats
        }


class ComprehensiveEvaluator:
    """Main evaluator that combines all evaluation components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize component evaluators
        self.synthetic_evaluator = SyntheticAttackEvaluator(config['metrics'])
        self.precision_evaluator = PrecisionAtKEvaluator(config['metrics'])
        self.latency_evaluator = LatencyEvaluator(config['metrics'])
        self.drift_monitor = DriftMonitor(config['monitoring']['drift_detection'])
        self.volume_analyzer = AlertVolumeAnalyzer(config)
        
    def run_comprehensive_evaluation(self, predictions_df: pd.DataFrame,
                                   campaigns_file: Optional[str] = None,
                                   weak_labels_df: Optional[pd.DataFrame] = None,
                                   baseline_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        
        results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions_df)
        }
        
        # 1. Synthetic attack recall evaluation
        if campaigns_file and Path(campaigns_file).exists():
            logger.info("Evaluating synthetic attack recall...")
            recall_results = self.synthetic_evaluator.evaluate_campaign_recall(
                predictions_df, campaigns_file
            )
            results['synthetic_attack_recall'] = recall_results
        else:
            logger.warning("No campaigns file provided, skipping synthetic recall evaluation")
            results['synthetic_attack_recall'] = {'error': 'No campaigns file'}
        
        # 2. Precision@K evaluation
        logger.info("Evaluating precision@K...")
        precision_results = self.precision_evaluator.evaluate_precision_at_k(
            predictions_df, weak_labels_df
        )
        results['precision_at_k'] = precision_results
        
        # 3. Detection latency evaluation
        if campaigns_file and Path(campaigns_file).exists():
            logger.info("Evaluating detection latency...")
            latency_results = self.latency_evaluator.evaluate_detection_latency(
                predictions_df, campaigns_file
            )
            results['detection_latency'] = latency_results
        else:
            results['detection_latency'] = {'error': 'No campaigns file'}
        
        # 4. Alert volume analysis
        logger.info("Analyzing alert volume...")
        volume_results = self.volume_analyzer.analyze_alert_volume(predictions_df)
        results['alert_volume'] = volume_results
        
        # 5. Drift monitoring
        if baseline_df is not None:
            logger.info("Monitoring for drift...")
            
            # Template drift
            template_drift = self.drift_monitor.evaluate_template_drift(
                predictions_df, baseline_df
            )
            results['template_drift'] = template_drift
            
            # Score drift
            if 'score_ml' in predictions_df.columns and 'score_ml' in baseline_df.columns:
                score_drift = self.drift_monitor.evaluate_score_drift(
                    predictions_df['score_ml'].values,
                    baseline_df['score_ml'].values
                )
                results['score_drift'] = score_drift
        else:
            logger.warning("No baseline data provided, skipping drift monitoring")
            results['template_drift'] = {'error': 'No baseline data'}
            results['score_drift'] = {'error': 'No baseline data'}
        
        # 6. Overall performance summary
        results['performance_summary'] = self._compute_performance_summary(results)
        
        return results
    
    def _compute_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall performance summary."""
        summary = {}
        
        # Recall performance
        recall_data = results.get('synthetic_attack_recall', {})
        if 'overall_recall' in recall_data:
            summary['recall_score'] = recall_data['overall_recall']
            summary['recall_meets_target'] = recall_data.get('meets_threshold', False)
        
        # Precision performance
        precision_data = results.get('precision_at_k', {})
        if 'precision_at_k' in precision_data:
            summary['precision_at_k_score'] = precision_data['precision_at_k']
            summary['precision_meets_target'] = precision_data['precision_at_k'] >= 0.6  # Target from requirements
        
        # Latency performance
        latency_data = results.get('detection_latency', {})
        if 'p95_latency' in latency_data:
            summary['p95_latency_seconds'] = latency_data['p95_latency']
            summary['latency_meets_target'] = latency_data.get('meets_p95_target', False)
        
        # Alert volume
        volume_data = results.get('alert_volume', {})
        if 'alerts_per_day' in volume_data:
            summary['alerts_per_day'] = volume_data['alerts_per_day']
            summary['alert_rate'] = volume_data['alert_rate']
        
        # Drift detection
        template_drift = results.get('template_drift', {})
        score_drift = results.get('score_drift', {})
        
        summary['drift_detected'] = (
            template_drift.get('template_drift_detected', False) or
            score_drift.get('score_drift_detected', False)
        )
        
        # Overall pass/fail
        meets_targets = []
        if 'recall_meets_target' in summary:
            meets_targets.append(summary['recall_meets_target'])
        if 'precision_meets_target' in summary:
            meets_targets.append(summary['precision_meets_target'])
        if 'latency_meets_target' in summary:
            meets_targets.append(summary['latency_meets_target'])
        
        if meets_targets:
            summary['all_targets_met'] = all(meets_targets)
            summary['targets_met_ratio'] = sum(meets_targets) / len(meets_targets)
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to {output_path}")
        
        # Also create a summary report
        summary_path = output_path.replace('.json', '_summary.txt')
        self._create_summary_report(results, summary_path)
    
    def _create_summary_report(self, results: Dict[str, Any], output_path: str):
        """Create human-readable summary report."""
        with open(output_path, 'w') as f:
            f.write("LogBERT Evaluation Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Timestamp: {results.get('evaluation_timestamp', 'N/A')}\n")
            f.write(f"Total Predictions: {results.get('total_predictions', 'N/A')}\n\n")
            
            # Performance summary
            summary = results.get('performance_summary', {})
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            
            if 'recall_score' in summary:
                f.write(f"Synthetic Attack Recall: {summary['recall_score']:.3f}")
                f.write(f" ({'✓' if summary.get('recall_meets_target', False) else '✗'})\n")
            
            if 'precision_at_k_score' in summary:
                f.write(f"Precision@K: {summary['precision_at_k_score']:.3f}")
                f.write(f" ({'✓' if summary.get('precision_meets_target', False) else '✗'})\n")
            
            if 'p95_latency_seconds' in summary:
                f.write(f"P95 Detection Latency: {summary['p95_latency_seconds']:.1f}s")
                f.write(f" ({'✓' if summary.get('latency_meets_target', False) else '✗'})\n")
            
            if 'alerts_per_day' in summary:
                f.write(f"Alert Volume: {summary['alerts_per_day']:.1f} alerts/day\n")
            
            if 'drift_detected' in summary:
                f.write(f"Drift Detected: {'Yes' if summary['drift_detected'] else 'No'}\n")
            
            if 'all_targets_met' in summary:
                f.write(f"\nAll Targets Met: {'Yes' if summary['all_targets_met'] else 'No'}\n")
        
        logger.info(f"Created summary report: {output_path}")


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config['evaluation'])
    
    # Load sample data (would come from actual predictions)
    try:
        # Mock predictions data for example
        predictions_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'source': np.random.choice(['192.168.1.1', '10.0.0.1', '203.1.1.1'], 1000),
            'score_ml': np.random.beta(2, 5, 1000)  # Skewed towards low scores
        })
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation(predictions_df)
        
        # Save results
        evaluator.save_results(results, './data/evaluation_results.json')
        
        print("Evaluation completed successfully!")
        print(f"Performance Summary:")
        summary = results.get('performance_summary', {})
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")