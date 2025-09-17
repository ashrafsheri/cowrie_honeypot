"""
Normalization module for Cowrie honeypot logs using Drain3 template mining.
Handles log parsing, template extraction, PII masking, and output normalization.
"""

import json
import pickle
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)


class PIIMasker:
    """Handles PII masking and bucketing for log normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port_buckets = config.get('port_buckets', [22, 80, 443, 1000, 5000, 10000, 65535])
        
        # Regex patterns for PII detection
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.user_pattern = re.compile(r'\b(?:user|username|login)[:=]\s*([a-zA-Z0-9_\-\.]+)', re.IGNORECASE)
        self.port_pattern = re.compile(r'\b(?:port|dst_port|src_port)[:=]\s*(\d+)', re.IGNORECASE)
        self.command_pattern = re.compile(r'\b(?:cmd|command)[:=]\s*["\']?([^"\';\n]+)', re.IGNORECASE)
        
        # Common SSH/system usernames to mask
        self.common_users = {
            'root', 'admin', 'administrator', 'user', 'test', 'guest',
            'oracle', 'postgres', 'mysql', 'www-data', 'ubuntu', 'centos'
        }
        
        # Common suspicious commands
        self.suspicious_commands = {
            'wget', 'curl', 'nc', 'nmap', 'chmod', 'su', 'sudo', 'passwd',
            'useradd', 'userdel', 'ps', 'netstat', 'ss', 'lsof', 'crontab'
        }
    
    def bucket_port(self, port: int) -> str:
        """Bucket ports into ranges for anonymization."""
        for bucket in self.port_buckets:
            if port <= bucket:
                return f"<PORT_{bucket}>"
        return "<PORT_HIGH>"
    
    def mask_ips(self, text: str) -> str:
        """Replace IP addresses with <IP> token."""
        if not self.config.get('mask_ips', True):
            return text
        return self.ip_pattern.sub('<IP>', text)
    
    def mask_users(self, text: str) -> str:
        """Replace usernames with <USER> token."""
        if not self.config.get('mask_users', True):
            return text
        
        # Mask explicit user fields
        text = self.user_pattern.sub(r'user=<USER>', text)
        
        # Mask common usernames appearing in text
        for user in self.common_users:
            text = re.sub(rf'\b{re.escape(user)}\b', '<USER>', text, flags=re.IGNORECASE)
        
        return text
    
    def mask_ports(self, text: str) -> str:
        """Replace ports with bucketed <PORT_X> tokens."""
        if not self.config.get('mask_ports', True):
            return text
        
        def replace_port(match):
            try:
                port = int(match.group(1))
                return match.group(0).replace(str(port), self.bucket_port(port))
            except (ValueError, IndexError):
                return match.group(0)
        
        return self.port_pattern.sub(replace_port, text)
    
    def mask_commands(self, text: str) -> str:
        """Replace suspicious commands with <CMD> token."""
        if not self.config.get('mask_commands', True):
            return text
        
        # Mask explicit command fields
        text = self.command_pattern.sub(r'cmd=<CMD>', text)
        
        # Mask suspicious commands in text
        for cmd in self.suspicious_commands:
            text = re.sub(rf'\b{re.escape(cmd)}\b', '<CMD>', text, flags=re.IGNORECASE)
        
        return text
    
    def mask_all(self, text: str) -> str:
        """Apply all PII masking rules."""
        text = self.mask_ips(text)
        text = self.mask_users(text) 
        text = self.mask_ports(text)
        text = self.mask_commands(text)
        return text


class CowrieDrainMiner:
    """Drain3-based template miner for Cowrie honeypot logs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.drain_config = config['drain3']
        self.pii_config = config.get('pii_masking', {})
        
        # Initialize PII masker
        self.pii_masker = PIIMasker(self.pii_config)
        
        # Initialize Drain3 template miner
        self._init_drain3()
        
        # Load persisted state if available
        self.template_cache_path = config.get('template_cache_path', './data/drain_state.pkl')
        self.save_frequency = config.get('save_frequency', 1000)
        self.processed_count = 0
        
        self._load_state()
        
    def _init_drain3(self):
        """Initialize Drain3 template miner with configuration."""
        config = TemplateMinerConfig()
        config.load({
            'DRAIN': {
                'sim_th': self.drain_config.get('sim_th', 0.4),
                'depth': self.drain_config.get('depth', 4), 
                'max_children': self.drain_config.get('max_children', 100),
                'max_clusters': self.drain_config.get('max_clusters', 1000),
                'extra_delimiters': self.drain_config.get('extra_delimiters', [])
            }
        })
        
        self.template_miner = TemplateMiner(config=config)
        
    def _load_state(self):
        """Load persisted Drain3 state if available."""
        if Path(self.template_cache_path).exists():
            try:
                with open(self.template_cache_path, 'rb') as f:
                    state = pickle.load(f)
                    self.template_miner.load_state(state['drain_state'])
                    self.processed_count = state.get('processed_count', 0)
                    logger.info(f"Loaded Drain3 state from {self.template_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load Drain3 state: {e}")
                
    def _save_state(self):
        """Save current Drain3 state."""
        try:
            Path(self.template_cache_path).parent.mkdir(parents=True, exist_ok=True)
            state = {
                'drain_state': self.template_miner.get_state(),
                'processed_count': self.processed_count
            }
            with open(self.template_cache_path, 'wb') as f:
                pickle.dump(state, f)
            logger.debug(f"Saved Drain3 state to {self.template_cache_path}")
        except Exception as e:
            logger.error(f"Failed to save Drain3 state: {e}")
    
    def extract_metadata(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from Cowrie log entry."""
        metadata = {
            'timestamp': log_entry.get('timestamp'),
            'source': log_entry.get('src_ip'),
            'session': log_entry.get('session'), 
            'sensor': log_entry.get('sensor'),
            'eventid': log_entry.get('eventid'),
            'severity': self._compute_severity(log_entry)
        }
        
        # Extract additional contextual fields
        for field in ['username', 'password', 'command', 'filename', 'protocol']:
            if field in log_entry:
                metadata[field] = log_entry[field]
                
        return metadata
    
    def _compute_severity(self, log_entry: Dict[str, Any]) -> str:
        """Compute severity level based on event type."""
        eventid = log_entry.get('eventid', '')
        
        # High severity events
        if eventid in ['cowrie.login.success', 'cowrie.command.success', 
                       'cowrie.session.file_download', 'cowrie.session.file_upload']:
            return 'high'
        
        # Medium severity events
        elif eventid in ['cowrie.login.failed', 'cowrie.command.failed']:
            return 'medium'
        
        # Low severity events (connections, disconnections)
        else:
            return 'low'
    
    def process_log_entry(self, log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single Cowrie log entry and extract template + parameters."""
        try:
            # Extract message content
            message = log_entry.get('message', '')
            if not message:
                return None
            
            # Apply PII masking
            masked_message = self.pii_masker.mask_all(message)
            
            # Extract template using Drain3
            result = self.template_miner.add_log_message(masked_message)
            template_id = result["cluster_id"]
            template = result["template_mined"]
            
            # Extract parameters (differences between original and template)
            parameters = self._extract_parameters(masked_message, template)
            
            # Extract metadata
            metadata = self.extract_metadata(log_entry)
            
            # Build normalized record
            normalized_record = {
                'timestamp': metadata['timestamp'],
                'source': metadata.get('source', 'unknown'),
                'host_id': metadata.get('sensor', 'unknown'),
                'session_id': metadata.get('session'),
                'template_id': template_id,
                'template': template,
                'parameters': json.dumps(parameters) if parameters else '{}',
                'severity': metadata['severity'],
                'eventid': metadata.get('eventid'),
                'raw_length': len(message),
                'masked_message': masked_message
            }
            
            self.processed_count += 1
            
            # Periodically save state
            if self.processed_count % self.save_frequency == 0:
                self._save_state()
                logger.info(f"Processed {self.processed_count} logs, "
                          f"found {len(self.template_miner.drain.clusters)} templates")
            
            return normalized_record
            
        except Exception as e:
            logger.error(f"Error processing log entry: {e}")
            return None
    
    def _extract_parameters(self, message: str, template: str) -> Dict[str, Any]:
        """Extract parameters by diffing message with template."""
        parameters = {}
        
        # Simple parameter extraction - replace wildcards with actual values
        message_tokens = message.split()
        template_tokens = template.split()
        
        if len(message_tokens) == len(template_tokens):
            param_idx = 0
            for msg_token, tmpl_token in zip(message_tokens, template_tokens):
                if tmpl_token == '<*>':
                    parameters[f'param_{param_idx}'] = msg_token
                    param_idx += 1
        
        return parameters
    
    def process_logs(self, logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process a batch of log entries."""
        normalized_records = []
        
        for log_entry in logs:
            normalized_record = self.process_log_entry(log_entry)
            if normalized_record:
                normalized_records.append(normalized_record)
        
        # Save final state
        self._save_state()
        
        # Convert to DataFrame
        df = pd.DataFrame(normalized_records)
        
        if not df.empty:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Normalized {len(normalized_records)} log entries into {df['template'].nunique() if not df.empty else 0} unique templates")
        
        return df
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered templates."""
        clusters = self.template_miner.drain.clusters
        
        stats = {
            'total_templates': len(clusters),
            'total_processed': self.processed_count,
            'top_templates': []
        }
        
        # Get top templates by frequency
        template_counts = {}
        for cluster_id, cluster in clusters.items():
            template_counts[cluster.get_template()] = cluster.size
        
        # Sort by frequency
        sorted_templates = sorted(template_counts.items(), 
                                key=lambda x: x[1], reverse=True)
        
        stats['top_templates'] = sorted_templates[:10]
        
        return stats


def load_cowrie_logs(file_path: str) -> List[Dict[str, Any]]:
    """Load Cowrie logs from JSONL file."""
    logs = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log_entry = json.loads(line)
                        logs.append(log_entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {e}")
    except FileNotFoundError:
        logger.error(f"Log file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading logs: {e}")
    
    return logs


def save_normalized_data(df: pd.DataFrame, output_path: str, format: str = 'parquet'):
    """Save normalized data in specified format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'parquet':
        df.to_parquet(f"{output_path}.parquet", index=False)
    elif format.lower() == 'csv':
        df.to_csv(f"{output_path}.csv", index=False)
    elif format.lower() == 'jsonl':
        df.to_json(f"{output_path}.jsonl", orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported output format: {format}")
    
    logger.info(f"Saved {len(df)} normalized records to {output_path}.{format}")


if __name__ == "__main__":
    # Example usage for testing
    import yaml
    
    # Load configuration
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize miner
    miner = CowrieDrainMiner(config['normalization'])
    
    # Load sample logs  
    logs = load_cowrie_logs('./data/sample/cowrie_sample.jsonl')
    
    if logs:
        # Process logs
        normalized_df = miner.process_logs(logs)
        
        # Save results
        save_normalized_data(normalized_df, './data/normalized/cowrie_normalized', 
                           config['dataset']['output_format'])
        
        # Print statistics
        stats = miner.get_template_stats()
        print(f"Template mining complete:")
        print(f"- Total templates: {stats['total_templates']}")
        print(f"- Total processed: {stats['total_processed']}")
        print(f"- Top templates:")
        for template, count in stats['top_templates'][:5]:
            print(f"  {count:4d}: {template}")
    else:
        print("No logs found to process")