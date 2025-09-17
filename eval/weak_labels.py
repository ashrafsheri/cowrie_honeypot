"""
Weak labeling system for generating pseudo-labels from rule-based heuristics.
No human labels available, so we use high-precision rules for evaluation.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class WeakLabelRule:
    """Base class for weak labeling rules."""
    
    def __init__(self, name: str, description: str, precision_estimate: float = 0.8):
        self.name = name
        self.description = description
        self.precision_estimate = precision_estimate
        
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply rule to dataset and return boolean series."""
        raise NotImplementedError
    
    def get_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Get confidence scores for rule application."""
        # Default: binary confidence based on rule match
        labels = self.apply(df)
        return labels.astype(float) * self.precision_estimate


class FailedLoginThresholdRule(WeakLabelRule):
    """Rule for detecting brute force attacks based on failed login frequency."""
    
    def __init__(self, threshold: int = 10, window_seconds: int = 60, precision: float = 0.9):
        super().__init__(
            name="failed_login_threshold",
            description=f"≥{threshold} failed logins per {window_seconds}s from same IP",
            precision_estimate=precision
        )
        self.threshold = threshold
        self.window_seconds = window_seconds
    
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply failed login threshold rule."""
        if df.empty:
            return pd.Series([], dtype=bool)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            logger.warning("No timestamp column found for failed login rule")
            return pd.Series([False] * len(df))
        
        # Find failed login templates
        failed_login_patterns = [
            'failed.*password',
            'login.*failed', 
            'authentication.*failed',
            'invalid.*user',
            'connection.*refused'
        ]
        
        failed_login_mask = pd.Series([False] * len(df))
        
        for pattern in failed_login_patterns:
            mask = df['template'].str.contains(pattern, case=False, na=False)
            failed_login_mask |= mask
        
        # Group by source IP and apply sliding window
        anomaly_mask = pd.Series([False] * len(df))
        
        if 'source' in df.columns:
            for source_ip, group in df.groupby('source'):
                source_failed_logins = group[failed_login_mask.loc[group.index]]
                
                if len(source_failed_logins) < self.threshold:
                    continue
                
                # Sort by timestamp
                source_failed_logins = source_failed_logins.sort_values('timestamp')
                
                # Apply sliding window
                for i in range(len(source_failed_logins)):
                    window_start = source_failed_logins.iloc[i]['timestamp']
                    window_end = window_start + timedelta(seconds=self.window_seconds)
                    
                    # Count failed logins in window
                    window_mask = (
                        (source_failed_logins['timestamp'] >= window_start) &
                        (source_failed_logins['timestamp'] <= window_end)
                    )
                    
                    window_count = window_mask.sum()
                    
                    if window_count >= self.threshold:
                        # Mark all sequences in this window as anomalous
                        window_indices = source_failed_logins[window_mask].index
                        anomaly_mask.loc[window_indices] = True
        
        return anomaly_mask


class SuspiciousCommandChainRule(WeakLabelRule):
    """Rule for detecting suspicious command sequences."""
    
    def __init__(self, suspicious_commands: List[str], chain_window: int = 300, precision: float = 0.8):
        super().__init__(
            name="suspicious_command_chain",
            description=f"Suspicious command chains within {chain_window}s",
            precision_estimate=precision
        )
        self.suspicious_commands = set(cmd.lower() for cmd in suspicious_commands)
        self.chain_window = chain_window
        
        # High-risk command combinations
        self.risky_chains = [
            {'wget', 'chmod'},
            {'curl', 'bash'},
            {'wget', 'sh'},
            {'nc', 'bash'},
            {'python', 'socket'},
            {'perl', 'socket'}
        ]
    
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply suspicious command chain rule."""
        if df.empty:
            return pd.Series([], dtype=bool)
        
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        anomaly_mask = pd.Series([False] * len(df))
        
        # Extract commands from templates
        df['extracted_commands'] = df['template'].apply(self._extract_commands)
        
        # Group by source and session for chain analysis
        groupby_cols = ['source']
        if 'session_id' in df.columns:
            groupby_cols.append('session_id')
        
        for group_key, group in df.groupby(groupby_cols):
            group = group.sort_values('timestamp') if 'timestamp' in group.columns else group
            
            # Sliding window analysis
            for i in range(len(group)):
                if 'timestamp' in group.columns:
                    window_start = group.iloc[i]['timestamp']
                    window_end = window_start + timedelta(seconds=self.chain_window)
                    
                    window_mask = (
                        (group['timestamp'] >= window_start) &
                        (group['timestamp'] <= window_end)
                    )
                    window_group = group[window_mask]
                else:
                    # Use fixed-size window if no timestamp
                    window_group = group.iloc[i:i+5]
                
                # Collect all commands in window
                window_commands = set()
                for commands in window_group['extracted_commands']:
                    window_commands.update(commands)
                
                # Check for risky command combinations
                is_risky = False
                for risky_chain in self.risky_chains:
                    if risky_chain.issubset(window_commands):
                        is_risky = True
                        break
                
                # Check for high count of suspicious commands
                suspicious_count = len(window_commands.intersection(self.suspicious_commands))
                if suspicious_count >= 3:
                    is_risky = True
                
                if is_risky:
                    anomaly_mask.loc[window_group.index] = True
        
        return anomaly_mask
    
    def _extract_commands(self, template: str) -> Set[str]:
        """Extract commands from template."""
        commands = set()
        
        # Common command extraction patterns
        patterns = [
            r'input=(\w+)',           # cowrie format
            r'command[:\s]+(\w+)',    # command: format
            r'executed:\s*(\w+)',     # executed: format
            r'^(\w+)\s',             # first word
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, template.lower())
            commands.update(matches)
        
        # Also check for direct command mentions
        for cmd in self.suspicious_commands:
            if cmd in template.lower():
                commands.add(cmd)
        
        return commands


class FileOperationAnomalyRule(WeakLabelRule):
    """Rule for detecting suspicious file operations."""
    
    def __init__(self, precision: float = 0.75):
        super().__init__(
            name="file_operation_anomaly",
            description="Suspicious file download/upload patterns",
            precision_estimate=precision
        )
        
        self.suspicious_extensions = {'.sh', '.py', '.pl', '.exe', '.bin', '.elf'}
        self.suspicious_domains = {'pastebin.com', 'bit.ly', 'tinyurl.com'}
    
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply file operation anomaly rule."""
        if df.empty:
            return pd.Series([], dtype=bool)
        
        anomaly_mask = pd.Series([False] * len(df))
        
        # File download patterns
        download_patterns = [
            r'file_download.*url=([^\s]+)',
            r'wget\s+([^\s]+)',
            r'curl.*?([http|ftp][^\s]+)'
        ]
        
        for pattern in download_patterns:
            matches = df['template'].str.extractall(pattern, flags=re.IGNORECASE)
            
            if not matches.empty:
                for idx, url in matches[0].items():
                    # Check for suspicious characteristics
                    is_suspicious = False
                    
                    # Suspicious file extensions
                    for ext in self.suspicious_extensions:
                        if ext in url.lower():
                            is_suspicious = True
                            break
                    
                    # Suspicious domains
                    for domain in self.suspicious_domains:
                        if domain in url.lower():
                            is_suspicious = True
                            break
                    
                    # IP addresses instead of domains (common in malware)
                    if re.match(r'https?://\d+\.\d+\.\d+\.\d+', url):
                        is_suspicious = True
                    
                    if is_suspicious:
                        anomaly_mask.iloc[idx[0]] = True
        
        # File execution after download
        df_with_commands = df[df['template'].str.contains('chmod|bash|sh|python', case=False, na=False)]
        
        for source, group in df_with_commands.groupby('source'):
            # Look for download followed by execution
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
                
                for i in range(len(group) - 1):
                    current = group.iloc[i]['template']
                    next_cmd = group.iloc[i + 1]['template']
                    
                    # Download followed by chmod/execution
                    if (('wget' in current.lower() or 'curl' in current.lower()) and
                        ('chmod' in next_cmd.lower() or 'bash' in next_cmd.lower())):
                        anomaly_mask.loc[group.iloc[i:i+2].index] = True
        
        return anomaly_mask


class NetworkAnomalyRule(WeakLabelRule):
    """Rule for detecting network-based anomalies."""
    
    def __init__(self, connection_threshold: int = 20, time_window: int = 60, precision: float = 0.7):
        super().__init__(
            name="network_anomaly",
            description=f"High connection frequency (≥{connection_threshold}/{time_window}s)",
            precision_estimate=precision
        )
        self.connection_threshold = connection_threshold
        self.time_window = time_window
    
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Apply network anomaly rule."""
        if df.empty:
            return pd.Series([], dtype=bool)
        
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        anomaly_mask = pd.Series([False] * len(df))
        
        # Connection-related templates
        connection_patterns = [
            'connection.*from',
            'new.*connection',
            'tcp.*connect',
            'port.*scan',
            'connect.*attempt'
        ]
        
        connection_mask = pd.Series([False] * len(df))
        for pattern in connection_patterns:
            mask = df['template'].str.contains(pattern, case=False, na=False)
            connection_mask |= mask
        
        connection_df = df[connection_mask]
        
        if connection_df.empty:
            return anomaly_mask
        
        # Group by source IP
        for source_ip, group in connection_df.groupby('source'):
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp')
                
                # Sliding window analysis
                for i in range(len(group)):
                    window_start = group.iloc[i]['timestamp']
                    window_end = window_start + timedelta(seconds=self.time_window)
                    
                    window_mask = (
                        (group['timestamp'] >= window_start) &
                        (group['timestamp'] <= window_end)
                    )
                    
                    window_connections = group[window_mask]
                    
                    if len(window_connections) >= self.connection_threshold:
                        anomaly_mask.loc[window_connections.index] = True
        
        return anomaly_mask


class WeakLabeler:
    """Main weak labeling system that combines multiple rules."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> List[WeakLabelRule]:
        """Initialize weak labeling rules based on configuration."""
        rules = []
        
        # Failed login threshold rule
        rules.append(FailedLoginThresholdRule(
            threshold=self.config.get('failed_login_threshold', 10),
            window_seconds=self.config.get('failed_login_window', 60)
        ))
        
        # Suspicious command chain rule
        suspicious_commands = self.config.get('suspicious_commands', 
                                            ['wget', 'curl', 'chmod', 'nmap', 'nc'])
        rules.append(SuspiciousCommandChainRule(
            suspicious_commands=suspicious_commands,
            chain_window=self.config.get('command_chain_window', 300)
        ))
        
        # File operation anomaly rule
        rules.append(FileOperationAnomalyRule())
        
        # Network anomaly rule
        rules.append(NetworkAnomalyRule(
            connection_threshold=20,
            time_window=60
        ))
        
        return rules
    
    def apply_weak_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all weak labeling rules to dataset."""
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Apply each rule
        rule_results = {}
        for rule in self.rules:
            try:
                rule_labels = rule.apply(df)
                rule_confidences = rule.get_confidence(df)
                
                rule_results[f"{rule.name}_label"] = rule_labels
                rule_results[f"{rule.name}_confidence"] = rule_confidences
                
                logger.info(f"Rule '{rule.name}' labeled {rule_labels.sum()} sequences as anomalous")
                
            except Exception as e:
                logger.error(f"Failed to apply rule '{rule.name}': {e}")
                rule_results[f"{rule.name}_label"] = pd.Series([False] * len(df))
                rule_results[f"{rule.name}_confidence"] = pd.Series([0.0] * len(df))
        
        # Add rule results to dataframe
        for col_name, series in rule_results.items():
            result_df[col_name] = series
        
        # Compute aggregate weak label
        label_columns = [col for col in rule_results.keys() if col.endswith('_label')]
        confidence_columns = [col for col in rule_results.keys() if col.endswith('_confidence')]
        
        if label_columns:
            # Any rule triggering makes it anomalous (high precision)
            result_df['weak_label'] = result_df[label_columns].any(axis=1)
            
            # Confidence is max of individual rule confidences
            result_df['weak_label_confidence'] = result_df[confidence_columns].max(axis=1)
            
            # Count of rules that triggered
            result_df['rules_triggered'] = result_df[label_columns].sum(axis=1)
        else:
            result_df['weak_label'] = False
            result_df['weak_label_confidence'] = 0.0
            result_df['rules_triggered'] = 0
        
        return result_df
    
    def get_label_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about weak labeling results."""
        if 'weak_label' not in df.columns:
            return {'error': 'No weak labels found. Run apply_weak_labels first.'}
        
        stats = {
            'total_sequences': len(df),
            'anomalous_sequences': df['weak_label'].sum(),
            'anomaly_rate': df['weak_label'].mean(),
            'avg_confidence': df['weak_label_confidence'].mean(),
            'rule_stats': {}
        }
        
        # Per-rule statistics
        label_columns = [col for col in df.columns if col.endswith('_label') and col != 'weak_label']
        
        for rule_col in label_columns:
            rule_name = rule_col.replace('_label', '')
            confidence_col = f"{rule_name}_confidence"
            
            if confidence_col in df.columns:
                rule_stats = {
                    'triggered_sequences': df[rule_col].sum(),
                    'trigger_rate': df[rule_col].mean(),
                    'avg_confidence': df[df[rule_col]][confidence_col].mean() if df[rule_col].any() else 0.0
                }
                stats['rule_stats'][rule_name] = rule_stats
        
        # Rules triggered distribution
        if 'rules_triggered' in df.columns:
            stats['rules_triggered_distribution'] = df['rules_triggered'].value_counts().to_dict()
        
        return stats


def apply_weak_labels_to_dataset(dataset_path: str, config: Dict[str, Any], 
                                output_path: Optional[str] = None) -> pd.DataFrame:
    """Apply weak labels to a dataset and optionally save results."""
    
    # Load dataset
    if dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.jsonl'):
        df = pd.read_json(dataset_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")
    
    # Initialize weak labeler
    labeler = WeakLabeler(config)
    
    # Apply weak labels
    labeled_df = labeler.apply_weak_labels(df)
    
    # Get statistics
    stats = labeler.get_label_stats(labeled_df)
    
    logger.info("Weak labeling completed:")
    logger.info(f"  Total sequences: {stats['total_sequences']}")
    logger.info(f"  Anomalous sequences: {stats['anomalous_sequences']} ({stats['anomaly_rate']:.2%})")
    logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
    
    # Save results if requested
    if output_path:
        labeled_df.to_parquet(output_path, index=False)
        
        # Save statistics
        stats_path = output_path.replace('.parquet', '_stats.json')
        with open(stats_path, 'w') as f:
            import json
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Saved labeled dataset to {output_path}")
        logger.info(f"Saved statistics to {stats_path}")
    
    return labeled_df


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    weak_label_config = config['evaluation']['weak_labels']
    
    # Apply weak labels to normalized dataset
    try:
        labeled_df = apply_weak_labels_to_dataset(
            './data/normalized/cowrie_normalized.parquet',
            weak_label_config,
            './data/labeled/cowrie_weak_labeled.parquet'
        )
        
        print(f"Successfully labeled {len(labeled_df)} sequences")
        print(f"Anomaly rate: {labeled_df['weak_label'].mean():.2%}")
        
    except FileNotFoundError:
        print("Normalized dataset not found. Run normalization first.")