"""
Enhanced Drain3 template mining module for SSH honeypot logs.
Specialized for Cowrie honeypot template patterns.
"""

import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import hashlib

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

# Import Cowrie preprocessor
from normalization.cowrie_preprocessor import CowriePreprocessor

logger = logging.getLogger(__name__)


class CowriePIIMasker:
    """Enhanced PII masker specialized for Cowrie honeypot logs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pii_config = config.get('normalization', {}).get('pii_masking', {})
        
        # Port bucketing
        self.port_buckets = config.get('normalization', {}).get('cowrie_patterns', {}).get(
            'port_buckets', [22, 80, 443, 1000, 5000, 10000, 65535]
        )
        
        # SSH-specific regex patterns
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.user_pattern = re.compile(r'\b(?:user(?:name)?|login)[:=]\s*([a-zA-Z0-9_\-\.]+)', re.IGNORECASE)
        self.port_pattern = re.compile(r'\b(?:port|dst_port|src_port)[:=]\s*(\d+)', re.IGNORECASE)
        self.password_pattern = re.compile(r'\b(?:pass(?:word)?|pwd)[:=]\s*([^\s,;"]+)', re.IGNORECASE)
        self.session_pattern = re.compile(r'\[session:\s*([a-f0-9]+)\]', re.IGNORECASE)
        
        # SSH command patterns
        self.cmd_pattern = re.compile(r'(?:input|command)[:=]\s*([^\n]+)', re.IGNORECASE)
        self.ssh_version_pattern = re.compile(r'SSH-\d+\.\d+-([^\s]+)')
        self.hassh_pattern = re.compile(r'hassh(?:Algorithms)?[:=]\s*([a-f0-9]+)', re.IGNORECASE)
        
        # Common SSH usernames to mask
        self.common_users = {
            'root', 'admin', 'administrator', 'user', 'test', 'guest',
            'support', 'oracle', 'postgres', 'mysql', 'ubuntu', 'ec2-user', 
            'centos', 'debian'
        }
        
        # Common SSH/honeypot commands
        self.common_commands = {
            'ls', 'cd', 'cat', 'wget', 'curl', 'uname', 'id', 'pwd',
            'chmod', 'mkdir', 'rm', 'cp', 'mv', 'bash', 'sh', 'ps',
            'netstat', 'ifconfig', 'system', 'python', 'perl', 'nc',
            'busybox'
        }
    
    def mask_message(self, message: str) -> str:
        """Apply all PII masking to a log message."""
        if not message:
            return message
            
        # Make sure message is a string
        if not isinstance(message, str):
            try:
                message = str(message)
            except Exception:
                return str(message) if message is not None else ""
            
        # Apply each masking function if enabled
        if self.pii_config.get('mask_ips', True):
            message = self.mask_ips(message)
            
        if self.pii_config.get('mask_users', True):
            message = self.mask_users(message)
            
        if self.pii_config.get('mask_ports', True):
            message = self.mask_ports(message)
            
        if self.pii_config.get('mask_passwords', True):
            message = self.mask_passwords(message)
            
        if self.pii_config.get('mask_sessions', True):
            message = self.mask_sessions(message)
            
        if self.pii_config.get('mask_commands', True):
            message = self.mask_commands(message)
            
        if self.pii_config.get('mask_hashes', True):
            message = self.mask_hashes(message)
            
        return message
    
    def mask_ips(self, text: str) -> str:
        """Replace IP addresses with <IP> token."""
        return self.ip_pattern.sub("<IP>", text)
    
    def mask_users(self, text: str) -> str:
        """Replace usernames with <USER> token."""
        # Replace explicit username patterns
        text = self.user_pattern.sub(r"\1=<USER>", text)
        
        # Replace common usernames
        for user in self.common_users:
            # Only replace whole words (with word boundaries)
            text = re.sub(rf'\b{re.escape(user)}\b', "<USER>", text)
            
        return text
    
    def mask_ports(self, text: str) -> str:
        """Replace port numbers with <PORT> token or bucketed value."""
        def replace_port(match):
            try:
                port = int(match.group(1))
                return f"port=<PORT_{self.bucket_port(port)}>"
            except ValueError:
                return f"port=<PORT>"
                
        return self.port_pattern.sub(replace_port, text)
    
    def bucket_port(self, port: int) -> str:
        """Bucket port numbers into ranges."""
        for i, bucket in enumerate(self.port_buckets):
            if port <= bucket:
                if i == 0:
                    return str(bucket)
                else:
                    return f"{self.port_buckets[i-1]+1}-{bucket}"
        return f"{self.port_buckets[-1]}+"
    
    def mask_passwords(self, text: str) -> str:
        """Replace password values with <PASSWORD> token."""
        return self.password_pattern.sub(r"\1=<PASSWORD>", text)
    
    def mask_sessions(self, text: str) -> str:
        """Replace session IDs with <SESSION> token."""
        return self.session_pattern.sub(r"[session: <SESSION>]", text)
    
    def mask_commands(self, text: str) -> str:
        """Replace command contents with <CMD> tokens."""
        def replace_cmd(match):
            cmd_str = match.group(1).strip()
            cmd_parts = cmd_str.split()
            
            if not cmd_parts:
                return "input=<CMD>"
                
            # Keep the command name but mask the arguments
            cmd_name = cmd_parts[0]
            if cmd_name in self.common_commands:
                return f"input={cmd_name} <CMD_ARGS>"
            else:
                return "input=<CMD>"
        
        return self.cmd_pattern.sub(replace_cmd, text)
    
    def mask_hashes(self, text: str) -> str:
        """Replace SSH fingerprints and hashes with tokens."""
        # Mask HASSH fingerprints
        text = self.hassh_pattern.sub(r"hassh=<HASSH>", text)
        
        # Mask SSH client versions
        text = self.ssh_version_pattern.sub(r"SSH-X.X-<SSH_VERSION>", text)
        
        return text


class CowrieDrainMiner:
    """Enhanced Drain3 log template miner for Cowrie honeypot logs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.norm_config = config.get('normalization', {})
        self.drain_config = self.norm_config.get('drain3', {})
        
        # Create the masker
        self.masker = CowriePIIMasker(config)
        
        # Create Drain3 configuration
        drain_conf = TemplateMinerConfig()
        drain_conf.drain_sim_th = self.drain_config.get('sim_th', 0.4)
        drain_conf.drain_depth = self.drain_config.get('depth', 5)
        drain_conf.drain_max_children = self.drain_config.get('max_children', 150)
        drain_conf.drain_max_clusters = self.drain_config.get('max_clusters', 2000)
        drain_conf.drain_extra_delimiters = self.drain_config.get(
            'extra_delimiters', ["=", ":", ",", ";", "|", "&", ">", "<"]
        )
        
        # Setup persistence
        cache_path = self.norm_config.get('template_cache_path', './data/drain_state.pkl')
        self.persistence = FilePersistence(cache_path)
        
        # Create template miner
        self.template_miner = TemplateMiner(self.persistence, drain_conf)
        
        # Create preprocessor
        self.preprocessor = CowriePreprocessor(config)
        
        data_config = config.get('data', {})
        self.source_field = data_config.get('source_field', 'src_ip')
        self.host_id_field = data_config.get('host_id_field', 'sensor')
        # Dictionary to store template stats
        self.template_stats = {}
        self.templates_by_id = {}
        self.template_clusters = {}
        self.event_templates = {}
        
        # Processed record count for periodic saving
        self.processed_count = 0
        self.save_frequency = self.norm_config.get('save_frequency', 1000)
        
    def process_log_files(self, log_paths: List[str]) -> pd.DataFrame:
        """Process multiple Cowrie log files and extract templates."""
        all_templates = []
        
        for log_path in log_paths:
            logger.info(f"Processing log file: {log_path}")
            
            # Parse the log file using Cowrie preprocessor
            df = self.preprocessor.parse_log_file(log_path)
            
            if df.empty:
                logger.warning(f"No data extracted from {log_path}")
                continue
                
            # Process each log record
            for _, row in df.iterrows():
                # Extract message and apply PII masking
                message = row.get('message', '')
                eventid = row.get('eventid', '')
                
                if not message:
                    continue
                
                # Apply PII masking
                masked_message = self.masker.mask_message(message)
                
                # Mine template with Drain3
                result = self.template_miner.add_log_message(masked_message)
                template_id = result["cluster_id"]
                template_text = result["template_mined"]
                
                # Update template stats
                if template_id not in self.template_stats:
                    self.template_stats[template_id] = {
                        'count': 0,
                        'event_types': set(),
                        'sources': set(),
                        'sessions': set()
                    }
                    
                self.template_stats[template_id]['count'] += 1
                self.template_stats[template_id]['event_types'].add(eventid)
                
                if 'src_ip' in row:
                    self.template_stats[template_id]['sources'].add(row['src_ip'])
                
                if 'session' in row:
                    self.template_stats[template_id]['sessions'].add(row['session'])
                
                # Store template mapping
                self.templates_by_id[template_id] = template_text
                
                # Track templates by event type
                if eventid not in self.event_templates:
                    self.event_templates[eventid] = set()
                self.event_templates[eventid].add(template_id)
                
                source_value = row.get(self.source_field) or row.get('src_ip') or \
                    row.get('sensor') or self.config.get('data', {}).get('source_name', 'cowrie')
                host_value = row.get(self.host_id_field) or row.get('sensor') or 'unknown'
                session_id = row.get('session')
                severity = self._compute_severity(eventid)
                parameters = self._extract_parameters(masked_message, template_text)
                parameters_json = json.dumps(parameters) if parameters else '{}'

                # Create template record
                template_record = {
                    'timestamp': row.get('timestamp'),
                    'template_id': template_id,
                    'template': template_text,
                    'eventid': eventid,
                    'event_category': row.get('event_category', 'other'),
                    'source': source_value,
                    'source_ip': row.get('src_ip'),
                    'host_id': host_value,
                    'session': session_id,
                    'session_id': session_id,
                    'attack_pattern': row.get('attack_pattern', False),
                    'severity': severity,
                    'parameters': parameters_json,
                    'original_message': message,
                    'masked_message': masked_message
                }

                all_templates.append(template_record)
                
                # Increment processed count and save periodically
                self.processed_count += 1
                if self.processed_count % self.save_frequency == 0:
                    logger.info(f"Processed {self.processed_count} records, saving state...")
                    # Let drain3 handle its own state saving
                    # We don't need to manually save
                
        # Save final state is automatically handled by Drain3
        # No need to manually save
        
        # Convert to DataFrame
        if not all_templates:
            logger.warning("No templates were extracted from the logs")
            return pd.DataFrame()
            
        templates_df = pd.DataFrame(all_templates)
        
        # Convert timestamp to datetime if present
        if 'timestamp' in templates_df.columns:
            templates_df['timestamp'] = pd.to_datetime(templates_df['timestamp'])

        return templates_df

    def _compute_severity(self, eventid: str) -> str:
        """Roughly bucket events by severity for downstream features."""
        if not eventid:
            return 'low'

        high_events = {
            'cowrie.login.success',
            'cowrie.command.success',
            'cowrie.session.file_download',
            'cowrie.session.file_upload'
        }
        medium_events = {
            'cowrie.login.failed',
            'cowrie.command.failed',
            'cowrie.session.closed'
        }

        if eventid in high_events:
            return 'high'
        if eventid in medium_events:
            return 'medium'
        return 'low'

    def _extract_parameters(self, message: str, template: str) -> Dict[str, Any]:
        """Extract wildcard parameters from Drain template output."""
        parameters: Dict[str, Any] = {}

        message_tokens = message.split()
        template_tokens = template.split()

        if len(message_tokens) == len(template_tokens):
            param_idx = 0
            for msg_token, tmpl_token in zip(message_tokens, template_tokens):
                if tmpl_token == '<*>':
                    parameters[f'param_{param_idx}'] = msg_token
                    param_idx += 1

        return parameters
        
    def save_results(self, output_path: str, templates_df: pd.DataFrame) -> None:
        """Save the template mining results."""
        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Save templates DataFrame
        templates_file = Path(output_path) / "templates.parquet"
        templates_df.to_parquet(templates_file)
        logger.info(f"Saved {len(templates_df)} templates to {templates_file}")
        
        # Save template stats
        stats_records = []
        for template_id, stats in self.template_stats.items():
            stats_records.append({
                'template_id': template_id,
                'template': self.templates_by_id.get(template_id, ''),
                'count': stats['count'],
                'event_types': list(stats['event_types']),
                'unique_sources': len(stats['sources']),
                'unique_sessions': len(stats['sessions'])
            })
        
        stats_df = pd.DataFrame(stats_records)
        stats_file = Path(output_path) / "template_stats.parquet"
        stats_df.to_parquet(stats_file)
        logger.info(f"Saved template stats for {len(stats_df)} templates to {stats_file}")
        
        # Save event type mapping
        event_templates_records = []
        for event_type, template_ids in self.event_templates.items():
            event_templates_records.append({
                'event_type': event_type,
                'template_count': len(template_ids),
                'template_ids': list(template_ids)
            })
        
        event_df = pd.DataFrame(event_templates_records)
        event_file = Path(output_path) / "event_templates.parquet"
        event_df.to_parquet(event_file)
        logger.info(f"Saved template mapping for {len(event_df)} event types to {event_file}")
        
        # Save drain3 configuration
        config_file = Path(output_path) / "drain_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'drain3': self.drain_config,
                'template_count': len(self.template_stats),
                'event_type_count': len(self.event_templates),
                'processed_records': self.processed_count
            }, f, indent=2)
        logger.info(f"Saved configuration to {config_file}")
