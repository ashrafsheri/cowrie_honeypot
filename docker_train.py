"""
Cowrie LogBERT Docker training script with MPS support.
For running training in Docker with Apple Silicon acceleration.
"""

import os
import sys
import yaml
import logging
import argparse
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(config):
    """Configure device for training (CUDA, MPS, or CPU)."""
    use_cuda = config.get('training', {}).get('use_cuda', True)
    use_mps = config.get('training', {}).get('use_mps', True)
    
    # Check available devices
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training (no CUDA or MPS available)")
    
    return device

def main():
    """Main entry point for Docker training."""
    parser = argparse.ArgumentParser(description="Cowrie LogBERT Training in Docker with MPS")
    parser.add_argument("--config", default="/home/app/configs/config.yaml", 
                        help="Path to config file")
    parser.add_argument("--skip-templates", action="store_true", 
                        help="Skip template extraction")
    parser.add_argument("--skip-dataset", action="store_true", 
                        help="Skip dataset building")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        sys.exit(1)
    
    # Setup device
    device = setup_device(config)
    logger.info(f"Using device: {device}")
    
    # Set environment variables for MPS/CUDA
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
    # Import training modules here to ensure correct device is set
    try:
        from train_cowrie import main as train_main
        logger.info("Starting Cowrie training pipeline")
        train_main()
    except ImportError as e:
        logger.error(f"Failed to import training modules: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()