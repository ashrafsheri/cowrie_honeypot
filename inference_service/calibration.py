"""
Score calibration for LogBERT anomaly detection.
Implements rolling percentile calibration and per-source calibration.
"""

import logging
import time
import json
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CalibrationStats:
    """Statistics for calibration monitoring."""
    total_scores: int = 0
    mean_raw_score: float = 0.0
    std_raw_score: float = 0.0
    mean_calibrated_score: float = 0.0
    percentile_25: float = 0.0
    percentile_50: float = 0.0  
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    last_updated: float = 0.0


class RollingBuffer:
    """Efficient rolling buffer for score tracking."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.sum = 0.0
        self.sum_sq = 0.0
    
    def add(self, value: float):
        """Add value to buffer."""
        if len(self.buffer) == self.max_size:
            # Remove oldest value from sums
            old_value = self.buffer[0]
            self.sum -= old_value
            self.sum_sq -= old_value ** 2
        
        # Add new value
        self.buffer.append(value)
        self.sum += value
        self.sum_sq += value ** 2
    
    def mean(self) -> float:
        """Get mean of buffer."""
        if not self.buffer:
            return 0.0
        return self.sum / len(self.buffer)
    
    def std(self) -> float:
        """Get standard deviation of buffer."""
        if len(self.buffer) < 2:
            return 0.0
        
        n = len(self.buffer)
        variance = (self.sum_sq / n) - (self.sum / n) ** 2
        return np.sqrt(max(variance, 0))
    
    def percentile(self, q: float) -> float:
        """Get percentile of buffer."""
        if not self.buffer:
            return 0.0
        return np.percentile(list(self.buffer), q)
    
    def to_array(self) -> np.ndarray:
        """Convert buffer to numpy array."""
        return np.array(list(self.buffer))


class RollingCalibrator:
    """Basic rolling percentile calibrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.window_size = config.get('window_size', 10000)
        self.percentile_window = config.get('percentile_window', 1000)
        self.update_frequency = config.get('update_frequency', 100)
        
        # Rolling score buffer
        self.score_buffer = RollingBuffer(self.window_size)
        
        # Percentile cache for efficiency
        self.percentile_cache = {}
        self.cache_update_count = 0
        
        # Statistics
        self.stats = CalibrationStats()
        
    def calibrate_score(self, raw_score: float, source: Optional[str] = None) -> float:
        """Calibrate raw score to [0, 1] range using rolling percentiles."""
        # Add to buffer
        self.score_buffer.add(raw_score)
        
        # Update cache periodically
        if self.cache_update_count % self.update_frequency == 0:
            self._update_percentile_cache()
        
        self.cache_update_count += 1
        
        # Apply percentile calibration
        if len(self.score_buffer.buffer) < 10:
            # Not enough data for calibration
            return 0.5
        
        # Use cached percentiles for efficiency
        p25 = self.percentile_cache.get('25', 0.0)
        p95 = self.percentile_cache.get('95', 1.0)
        
        # Linear scaling between p25 (score=0.1) and p95 (score=0.9)
        if p95 > p25:
            calibrated = 0.1 + 0.8 * (raw_score - p25) / (p95 - p25)
        else:
            calibrated = 0.5
        
        # Clamp to [0, 1]
        calibrated = max(0.0, min(1.0, calibrated))
        
        # Update statistics
        self._update_stats(raw_score, calibrated)
        
        return calibrated
    
    def _update_percentile_cache(self):
        """Update percentile cache for efficiency."""
        if len(self.score_buffer.buffer) >= 10:
            for p in [25, 50, 75, 95]:
                self.percentile_cache[str(p)] = self.score_buffer.percentile(p)
    
    def _update_stats(self, raw_score: float, calibrated_score: float):
        """Update calibration statistics."""
        self.stats.total_scores += 1
        self.stats.last_updated = time.time()
        
        # Update means (exponential moving average)
        alpha = 0.01
        self.stats.mean_raw_score = (1 - alpha) * self.stats.mean_raw_score + alpha * raw_score
        self.stats.mean_calibrated_score = (1 - alpha) * self.stats.mean_calibrated_score + alpha * calibrated_score
        
        # Update percentile stats from cache
        if self.percentile_cache:
            self.stats.percentile_25 = self.percentile_cache.get('25', 0.0)
            self.stats.percentile_50 = self.percentile_cache.get('50', 0.0)
            self.stats.percentile_75 = self.percentile_cache.get('75', 0.0)
            self.stats.percentile_95 = self.percentile_cache.get('95', 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return asdict(self.stats)


class PerSourceCalibrator:
    """Per-source calibration for better accuracy."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.per_source_enabled = config.get('per_source_calibration', True)
        self.min_samples_per_source = config.get('min_samples_per_source', 100)
        
        # Per-source calibrators
        self.source_calibrators: Dict[str, RollingCalibrator] = {}
        
        # Global fallback calibrator
        self.global_calibrator = RollingCalibrator(config)
        
        # Source statistics
        self.source_stats = defaultdict(int)
        
        # Persistence
        self.state_path = config.get('state_path', './data/calibration_state.pkl')
        self._load_state()
    
    def calibrate_score(self, raw_score: float, source: str) -> float:
        """Calibrate score with per-source adaptation."""
        self.source_stats[source] += 1
        
        if not self.per_source_enabled:
            return self.global_calibrator.calibrate_score(raw_score)
        
        # Use per-source calibrator if we have enough samples
        if (source in self.source_calibrators and 
            self.source_stats[source] >= self.min_samples_per_source):
            
            # Use source-specific calibrator
            calibrated = self.source_calibrators[source].calibrate_score(raw_score)
            
            # Also update global calibrator for fallback
            self.global_calibrator.calibrate_score(raw_score)
            
        else:
            # Initialize or update source calibrator
            if source not in self.source_calibrators:
                self.source_calibrators[source] = RollingCalibrator(self.config)
            
            self.source_calibrators[source].calibrate_score(raw_score)
            
            # Use global calibrator for scoring
            calibrated = self.global_calibrator.calibrate_score(raw_score)
        
        # Periodically save state
        if sum(self.source_stats.values()) % 1000 == 0:
            self._save_state()
        
        return calibrated
    
    def _save_state(self):
        """Save calibration state for persistence."""
        try:
            Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'source_stats': dict(self.source_stats),
                'global_stats': self.global_calibrator.get_stats(),
                'source_calibrator_stats': {
                    src: cal.get_stats() 
                    for src, cal in self.source_calibrators.items()
                }
            }
            
            with open(self.state_path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.debug(f"Saved calibration state to {self.state_path}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration state: {e}")
    
    def _load_state(self):
        """Load calibration state if available."""
        try:
            if Path(self.state_path).exists():
                with open(self.state_path, 'rb') as f:
                    state = pickle.load(f)
                
                self.source_stats.update(state.get('source_stats', {}))
                logger.info(f"Loaded calibration state from {self.state_path}")
                logger.info(f"Tracking {len(self.source_stats)} sources")
                
        except Exception as e:
            logger.warning(f"Failed to load calibration state: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive calibration statistics."""
        return {
            'global_stats': self.global_calibrator.get_stats(),
            'total_sources': len(self.source_stats),
            'sources_with_enough_samples': sum(
                1 for count in self.source_stats.values() 
                if count >= self.min_samples_per_source
            ),
            'top_sources': dict(
                sorted(self.source_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            'per_source_enabled': self.per_source_enabled,
            'min_samples_per_source': self.min_samples_per_source
        }
    
    def get_source_stats(self, source: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific source."""
        if source in self.source_calibrators:
            return {
                'sample_count': self.source_stats[source],
                'calibration_stats': self.source_calibrators[source].get_stats(),
                'using_source_calibrator': self.source_stats[source] >= self.min_samples_per_source
            }
        return None


class IsotonicCalibrator:
    """Isotonic regression calibrator (placeholder for future implementation)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.warning("IsotonicCalibrator not implemented, falling back to RollingCalibrator")
        self.fallback = RollingCalibrator(config)
    
    def calibrate_score(self, raw_score: float, source: Optional[str] = None) -> float:
        """Calibrate using isotonic regression (fallback to rolling for now)."""
        return self.fallback.calibrate_score(raw_score, source)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        return self.fallback.get_stats()


def create_calibrator(config: Dict[str, Any]) -> RollingCalibrator:
    """Factory function to create appropriate calibrator."""
    method = config.get('method', 'rolling_percentile')
    
    if method == 'rolling_percentile':
        if config.get('per_source_calibration', False):
            return PerSourceCalibrator(config)
        else:
            return RollingCalibrator(config)
    elif method == 'isotonic':
        return IsotonicCalibrator(config)
    else:
        logger.warning(f"Unknown calibration method: {method}, using rolling_percentile")
        return RollingCalibrator(config)


if __name__ == "__main__":
    # Test calibrator
    import random
    
    config = {
        'window_size': 1000,
        'percentile_window': 100,
        'update_frequency': 50,
        'per_source_calibration': True,
        'min_samples_per_source': 50
    }
    
    calibrator = PerSourceCalibrator(config)
    
    # Generate test data
    sources = ['source_A', 'source_B', 'source_C']
    
    for i in range(500):
        source = random.choice(sources)
        
        # Generate raw scores (higher for source_C to simulate more anomalous)
        if source == 'source_C':
            raw_score = random.uniform(5.0, 10.0)
        else:
            raw_score = random.uniform(0.0, 5.0)
        
        calibrated_score = calibrator.calibrate_score(raw_score, source)
        
        if i % 100 == 0:
            print(f"Step {i}: Raw={raw_score:.2f}, Calibrated={calibrated_score:.2f}, Source={source}")
    
    # Print final statistics
    print("\nFinal Statistics:")
    stats = calibrator.get_stats()
    print(json.dumps(stats, indent=2))