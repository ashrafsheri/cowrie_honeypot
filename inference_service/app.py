"""
FastAPI inference service for LogBERT anomaly detection.
Provides real-time scoring with batching and calibration.
"""

import asyncio
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

try:
    from .calibration import RollingCalibrator, PerSourceCalibrator
except ImportError:
    from inference_service.calibration import RollingCalibrator, PerSourceCalibrator

logger = logging.getLogger(__name__)


# Request/Response Models
class LogSequence(BaseModel):
    """Single log sequence for scoring."""
    templates: List[str] = Field(..., description="List of log templates")
    source: str = Field(..., description="Source identifier (IP, host, etc.)")
    host_id: Optional[str] = Field(None, description="Host identifier")
    timestamp: Optional[str] = Field(None, description="Sequence timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")


class BatchScoringRequest(BaseModel):
    """Batch scoring request."""
    sequences: List[LogSequence] = Field(..., description="List of log sequences to score")
    return_explanations: bool = Field(False, description="Whether to include explanations")
    use_calibration: bool = Field(True, description="Whether to apply score calibration")


class ScoringResult(BaseModel):
    """Scoring result for a single sequence."""
    score_ml: float = Field(..., description="ML-based anomaly score [0-1]")
    score_raw: float = Field(..., description="Raw model score (NLL or distance)")
    source: str = Field(..., description="Source identifier")
    host_id: Optional[str] = Field(None, description="Host identifier")
    sequence_length: int = Field(..., description="Number of templates in sequence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Optional explanation")


class BatchScoringResponse(BaseModel):
    """Batch scoring response."""
    results: List[ScoringResult] = Field(..., description="Scoring results")
    batch_size: int = Field(..., description="Number of sequences processed")
    total_processing_time_ms: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    total_requests: int = Field(..., description="Total requests processed")
    avg_latency_ms: float = Field(..., description="Average latency")


# Core Inference Classes
class LogBertInference:
    """LogBERT model inference wrapper."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.vocab = None
        
        self._load_model()
        
        # Scoring configuration
        self.scoring_method = config.get('primary_method', 'mlm_nll')
        self.max_sequence_length = config.get('max_sequence_length', 256)
        
        # Performance tracking
        self.total_requests = 0
        self.total_latency = 0.0
        
    def _load_model(self):
        """Load the trained LogBERT model."""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.warning(f"Model path {self.model_path} does not exist. Running in demo mode with fallback model.")
                # Use a fallback model for demonstration
                self.model_path = 'distilbert-base-uncased'
            
            logger.info(f"Loading LogBERT model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model 
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load vocabulary if available
            vocab_path = os.path.join(self.model_path, 'template_vocab.json')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r') as f:
                    self.vocab = json.load(f)
                logger.info(f"Loaded template vocabulary with {len(self.vocab)} tokens")
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_sequences(self, sequences: List[LogSequence]) -> Dict[str, torch.Tensor]:
        """Preprocess log sequences for inference."""
        # Convert templates to text
        texts = []
        for seq in sequences:
            # Join templates with special separator
            text = " [SEP] ".join(seq.templates)
            texts.append(text)
        
        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def compute_mlm_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Compute anomaly score using MLM negative log-likelihood."""
        with torch.no_grad():
            # Get model predictions
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            # Add MLM head (simple linear layer for demo)
            mlm_logits = F.linear(hidden_states, self.model.embeddings.word_embeddings.weight)
            
            # Compute negative log-likelihood
            log_probs = F.log_softmax(mlm_logits, dim=-1)
            nll_scores = []
            
            for i in range(input_ids.size(0)):
                seq_input_ids = input_ids[i]
                seq_attention_mask = attention_mask[i]
                seq_log_probs = log_probs[i]
                
                # Compute NLL for non-padding tokens
                valid_positions = (seq_attention_mask == 1) & (seq_input_ids != self.tokenizer.pad_token_id)
                
                if valid_positions.sum() > 0:
                    # Get log probabilities for actual tokens
                    token_log_probs = seq_log_probs[valid_positions, seq_input_ids[valid_positions]]
                    nll = -token_log_probs.mean().item()
                else:
                    nll = 0.0
                
                nll_scores.append(nll)
            
            return np.array(nll_scores)
    
    def compute_embedding_distance(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Compute anomaly score using embedding distance to centroid."""
        with torch.no_grad():
            # Get embeddings
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            pooled_embeddings = (embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            
            # Compute distances (placeholder - would use rolling centroid in practice)
            # For now, compute distance to mean embedding
            centroid = pooled_embeddings.mean(dim=0, keepdim=True)
            distances = F.cosine_similarity(pooled_embeddings, centroid, dim=1)
            
            # Convert similarity to distance
            distances = 1 - distances
            
            return distances.cpu().numpy()
    
    def predict_batch(self, sequences: List[LogSequence]) -> Tuple[np.ndarray, float]:
        """Predict anomaly scores for a batch of sequences."""
        start_time = time.time()
        
        # Preprocess
        inputs = self.preprocess_sequences(sequences)
        
        # Compute scores based on method
        if self.scoring_method == 'mlm_nll':
            raw_scores = self.compute_mlm_score(inputs['input_ids'], inputs['attention_mask'])
        elif self.scoring_method == 'embedding_distance':
            raw_scores = self.compute_embedding_distance(inputs['input_ids'], inputs['attention_mask'])
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update performance tracking
        self.total_requests += len(sequences)
        self.total_latency += processing_time
        
        return raw_scores, processing_time
    
    def get_explanation(self, sequence: LogSequence, raw_score: float) -> Dict[str, Any]:
        """Generate explanation for anomaly score."""
        # Simplified explanation - in practice would use attention weights, 
        # nearest neighbors, etc.
        explanation = {
            'scoring_method': self.scoring_method,
            'raw_score': raw_score,
            'sequence_length': len(sequence.templates),
            'template_diversity': len(set(sequence.templates)),
            'suspicious_templates': []
        }
        
        # Find potentially suspicious templates (placeholder logic)
        suspicious_keywords = ['fail', 'error', 'attack', 'intrusion', 'wget', 'curl']
        for template in sequence.templates:
            if any(keyword in template.lower() for keyword in suspicious_keywords):
                explanation['suspicious_templates'].append(template)
        
        return explanation
    
    @property
    def avg_latency(self) -> float:
        """Get average latency per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests


class BatchProcessor:
    """Handles batching of incoming requests for throughput optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_batch_size = config.get('max_batch_size', 16)
        self.batch_timeout_ms = config.get('batch_timeout_ms', 50)
        self.max_queue_size = config.get('max_queue_size', 1000)
        
        # Batching queues
        self.pending_requests = deque()
        self.batch_lock = asyncio.Lock()
        
        # Background task for processing batches
        self.processing_task = None
        
    async def add_request(self, sequences: List[LogSequence], future: asyncio.Future):
        """Add request to batch queue."""
        async with self.batch_lock:
            if len(self.pending_requests) >= self.max_queue_size:
                raise HTTPException(status_code=503, detail="Request queue full")
            
            self.pending_requests.append((sequences, future))
    
    async def process_batches(self, inference_engine: LogBertInference, 
                            calibrator: RollingCalibrator):
        """Background task to process batches."""
        while True:
            try:
                batch_sequences = []
                batch_futures = []
                
                # Collect batch
                async with self.batch_lock:
                    while (len(batch_sequences) < self.max_batch_size and 
                           len(self.pending_requests) > 0):
                        sequences, future = self.pending_requests.popleft()
                        batch_sequences.extend(sequences)
                        batch_futures.append((future, len(sequences)))
                
                if batch_sequences:
                    # Process batch
                    raw_scores, processing_time = inference_engine.predict_batch(batch_sequences)
                    
                    # Apply calibration
                    calibrated_scores = []
                    for i, seq in enumerate(batch_sequences):
                        calibrated_score = calibrator.calibrate_score(
                            raw_scores[i], seq.source
                        )
                        calibrated_scores.append(calibrated_score)
                    
                    # Return results to futures
                    score_idx = 0
                    for future, seq_count in batch_futures:
                        seq_scores = calibrated_scores[score_idx:score_idx + seq_count]
                        seq_raw_scores = raw_scores[score_idx:score_idx + seq_count]
                        
                        if not future.done():
                            future.set_result((seq_scores, seq_raw_scores, processing_time))
                        
                        score_idx += seq_count
                
                # Wait before next batch
                await asyncio.sleep(self.batch_timeout_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(0.1)


# FastAPI Application
def create_app(config_path: str = './configs/config.yaml') -> FastAPI:
    """Create FastAPI application."""
    import yaml
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        inference_config = config['inference']
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Config file not found or incomplete: {e}. Using defaults.")
        inference_config = {
            'model_path': './models/logbert-mlm',
            'scoring': {'primary_method': 'mlm_nll'},
            'calibration': {'method': 'rolling_percentile'}
        }
    
    model_path = inference_config.get('model_path', './models/logbert-mlm')
    
    app = FastAPI(
        title="LogBERT Anomaly Detection API",
        description="Real-time anomaly detection for log sequences using LogBERT",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    inference_engine = LogBertInference(model_path, inference_config['scoring'])
    calibrator = PerSourceCalibrator(inference_config['calibration'])
    batch_processor = BatchProcessor(inference_config)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize background processing."""
        batch_processor.processing_task = asyncio.create_task(
            batch_processor.process_batches(inference_engine, calibrator)
        )
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup background tasks."""
        if batch_processor.processing_task:
            batch_processor.processing_task.cancel()
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=inference_engine.model is not None,
            total_requests=inference_engine.total_requests,
            avg_latency_ms=inference_engine.avg_latency
        )
    
    @app.post("/score", response_model=BatchScoringResponse)
    async def score_sequences(request: BatchScoringRequest):
        """Score log sequences for anomalies."""
        if not request.sequences:
            raise HTTPException(status_code=400, detail="No sequences provided")
        
        start_time = time.time()
        
        # Create future for batch processing
        future = asyncio.Future()
        await batch_processor.add_request(request.sequences, future)
        
        # Wait for results
        try:
            calibrated_scores, raw_scores, processing_time = await future
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
        
        # Build response
        results = []
        for i, (seq, cal_score, raw_score) in enumerate(
            zip(request.sequences, calibrated_scores, raw_scores)
        ):
            # Generate explanation if requested
            explanation = None
            if request.return_explanations:
                explanation = inference_engine.get_explanation(seq, raw_score)
            
            result = ScoringResult(
                score_ml=float(cal_score),
                score_raw=float(raw_score),
                source=seq.source,
                host_id=seq.host_id,
                sequence_length=len(seq.templates),
                processing_time_ms=processing_time / len(request.sequences),
                explanation=explanation
            )
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchScoringResponse(
            results=results,
            batch_size=len(request.sequences),
            total_processing_time_ms=total_time
        )
    
    @app.get("/stats")
    async def get_stats():
        """Get service statistics."""
        return {
            "model_path": model_path,
            "device": str(inference_engine.device),
            "scoring_method": inference_engine.scoring_method,
            "total_requests": inference_engine.total_requests,
            "avg_latency_ms": inference_engine.avg_latency,
            "calibrator_stats": calibrator.get_stats()
        }
    
    return app


def main():
    """Main function to run the service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LogBERT Inference Service')
    parser.add_argument('--config', default='./configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Create app
    app = create_app(args.config)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


# Create default app instance for uvicorn
app = create_app()

if __name__ == "__main__":
    main()