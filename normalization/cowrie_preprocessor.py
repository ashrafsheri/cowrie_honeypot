"""
Cowrie Honeypot log preprocessing module.
Handles specialized parsing and structuring for SSH/Telnet honeypot logs.
"""

import json
import re
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class CowriePreprocessor:
    """
    Specialized preprocessor for Cowrie honeypot logs.
    - Handles session tracking
    - Extracts command sequences
    - Identifies attack patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Known Cowrie event types for better categorization
        self.event_categories = {
            "connection": ["cowrie.session.connect", "cowrie.session.closed"],
            "authentication": [
                "cowrie.login.success", "cowrie.login.failed", 
                "cowrie.password.input", "cowrie.auth.success"
            ],
            "client_info": [
                "cowrie.client.version", "cowrie.client.kex",
                "cowrie.client.size", "cowrie.client.var"
            ],
            "commands": [
                "cowrie.command.input", "cowrie.command.success", 
                "cowrie.command.failed"
            ],
            "filesystem": [
                "cowrie.session.file_download", "cowrie.session.file_upload",
                "cowrie.session.input", "cowrie.session.output"
            ]
        }
        
        # Common attack pattern markers (commands, tools, behavior)
        self.attack_markers = [
            # Reconnaissance
            'uname -a', 'cat /proc/cpuinfo', 'free -m', 'lscpu',
            # Credential access
            'cat /etc/passwd', 'cat /etc/shadow',
            # Persistence
            'crontab -e', 'wget', 'curl', '.bashrc', 'rc.local',
            # Malware
            '.sh', 'chmod +x', 'python', 'perl', 'tftp', 'botnet',
            # Privilege escalation
            'sudo', 'su root', 'chmod 777', 'chown root'
        ]
    
    def parse_log_file(self, file_path: str) -> pd.DataFrame:
        """Parse a Cowrie JSON log file into a structured DataFrame."""
        logger.info(f"Parsing Cowrie log file: {file_path}")
        
        # Extract date from filename if possible
        file_date = None
        date_pattern = self.config['data']['files'].get('date_pattern')
        date_regex = self.config['data']['files'].get('date_extraction_regex')
        
        if date_regex:
            file_name = Path(file_path).name
            match = re.search(date_regex, file_name)
            if match and date_pattern:
                try:
                    file_date = datetime.strptime(match.group(1), date_pattern).date()
                    logger.debug(f"Extracted date from filename: {file_date}")
                except ValueError:
                    logger.warning(f"Failed to parse date from filename: {file_name}")
        
        # Parse log records
        records = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    # Add metadata
                    record['_file'] = Path(file_path).name
                    record['_line'] = line_num
                    record['_file_date'] = file_date
                    records.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num} in {file_path}")
                    continue
        
        if not records:
            logger.warning(f"No valid records found in {file_path}")
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        
        # Ensure timestamp field exists and convert to datetime
        timestamp_field = self.config['data'].get('timestamp_field', 'timestamp')
        if timestamp_field in df.columns:
            df[timestamp_field] = pd.to_datetime(df[timestamp_field])
        else:
            logger.warning(f"Timestamp field '{timestamp_field}' not found in log data")
            
        # Add categorization
        df['event_category'] = df['eventid'].apply(self.categorize_event)
        
        # Add attack pattern detection
        if 'message' in df.columns:
            df['attack_pattern'] = df['message'].apply(self.detect_attack_pattern)
            
        return df
    
    def categorize_event(self, event_id: str) -> str:
        """Categorize Cowrie event IDs into functional groups."""
        for category, events in self.event_categories.items():
            if event_id in events:
                return category
        return "other"
    
    def detect_attack_pattern(self, message: Optional[str]) -> bool:
        """Detect potential attack patterns in log messages."""
        if not message:
            return False
            
        # Ensure message is a string
        if not isinstance(message, str):
            try:
                message = str(message)
            except:
                return False
            
        message = message.lower()
        for marker in self.attack_markers:
            if marker.lower() in message:
                return True
        return False
    
    def group_by_session(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group log entries by Cowrie session ID for sequence analysis."""
        session_field = self.config['data'].get('session_field', 'session')
        
        if session_field not in df.columns:
            logger.warning(f"Session field '{session_field}' not found in log data")
            return {}
            
        # Group by session ID
        session_groups = {}
        for session_id, group in df.groupby(session_field):
            if not pd.isna(session_id) and session_id:
                # Sort by timestamp
                timestamp_field = self.config['data'].get('timestamp_field', 'timestamp')
                if timestamp_field in group.columns:
                    group = group.sort_values(timestamp_field)
                session_groups[session_id] = group
                
        return session_groups
    
    def extract_login_attempts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract login attempt information from log data."""
        login_records = []
        
        # Filter to authentication-related events
        auth_events = df[df['eventid'].isin([
            'cowrie.login.success', 'cowrie.login.failed',
            'cowrie.password.input', 'cowrie.auth.success'
        ])]
        
        # Group by session
        sessions = auth_events.groupby('session')
        
        for session_id, session_df in sessions:
            # Sort by timestamp
            session_df = session_df.sort_values('timestamp')
            
            # Extract username and password if available
            username = None
            password = None
            success = False
            
            for _, row in session_df.iterrows():
                event = row['eventid']
                
                if event == 'cowrie.login.success':
                    username = row.get('username')
                    success = True
                elif event == 'cowrie.password.input':
                    password = row.get('password')
                
            # Create record
            if username or password:
                login_records.append({
                    'session_id': session_id,
                    'timestamp': session_df['timestamp'].min(),
                    'username': username,
                    'password': password,
                    'success': success
                })
                
        return pd.DataFrame(login_records)
    
    def extract_command_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract command sequences from Cowrie logs."""
        cmd_records = []
        
        # Filter to command-related events
        cmd_events = df[df['eventid'].isin([
            'cowrie.command.input', 'cowrie.command.success', 
            'cowrie.command.failed'
        ])]
        
        # Group by session
        sessions = cmd_events.groupby('session')
        
        for session_id, session_df in sessions:
            # Sort by timestamp
            session_df = session_df.sort_values('timestamp')
            
            # Extract command sequence
            cmd_sequence = []
            
            for _, row in session_df.iterrows():
                if row['eventid'] == 'cowrie.command.input':
                    cmd = row.get('input', '')
                    if cmd:
                        cmd_sequence.append(cmd)
            
            if cmd_sequence:
                cmd_records.append({
                    'session_id': session_id,
                    'timestamp': session_df['timestamp'].min(),
                    'command_count': len(cmd_sequence),
                    'command_sequence': cmd_sequence
                })
                
        return pd.DataFrame(cmd_records)