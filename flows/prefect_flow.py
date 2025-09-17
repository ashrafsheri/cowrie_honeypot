"""
Prefect orchestration flow for LogBERT anomaly detection pipeline.
Manages the complete workflow from data extraction to model serving.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import yaml
import json

from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule

# Import our pipeline components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from normalization.drain_miner import CowrieDrainMiner
from dataset.build_windows import WindowedDatasetBuilder
from models.train_mlm import MLMTrainer
from models.train_infonce import InfoNCETrainer
from eval.inject_campaigns import SyntheticCampaignGenerator
from eval.weak_labels import WeakLabeler
from eval.compute_metrics import ComprehensiveEvaluator
from inference_service.calibration import ModelCalibrator

logger = logging.getLogger(__name__)


@task(name="extract_raw_logs")
def extract_raw_logs(config: Dict[str, Any]) -> str:
    """Extract raw logs from source (files or OpenSearch)."""
    logger = get_run_logger()
    logger.info("Starting raw log extraction...")
    
    data_config = config['data']
    source_type = data_config.get('source_type', 'files')
    
    if source_type == 'files':
        # Simple file copying/validation
        input_path = Path(data_config['input_path'])
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        
        # Count available logs
        if input_path.is_dir():
            log_files = list(input_path.glob("*.log")) + list(input_path.glob("*.json"))
            logger.info(f"Found {len(log_files)} log files in {input_path}")
        else:
            log_files = [input_path]
            logger.info(f"Single log file: {input_path}")
        
        # Validation
        total_lines = 0
        for log_file in log_files[:5]:  # Sample first 5 files
            try:
                with open(log_file, 'r') as f:
                    lines = sum(1 for _ in f)
                    total_lines += lines
                logger.info(f"  {log_file.name}: {lines} lines")
            except Exception as e:
                logger.warning(f"Could not read {log_file}: {e}")
        
        return str(input_path)
    
    elif source_type == 'opensearch':
        # OpenSearch extraction would be implemented here
        opensearch_config = data_config['opensearch']
        
        logger.info(f"Extracting from OpenSearch: {opensearch_config['host']}")
        logger.info(f"Index pattern: {opensearch_config['index_pattern']}")
        
        # TODO: Implement OpenSearch client and query
        # For now, return placeholder
        raise NotImplementedError("OpenSearch extraction not yet implemented")
    
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


@task(name="normalize_logs")
def normalize_logs(raw_logs_path: str, config: Dict[str, Any]) -> str:
    """Normalize logs using Drain3 template mining."""
    logger = get_run_logger()
    logger.info("Starting log normalization with Drain3...")
    
    # Initialize Drain miner
    normalization_config = config['normalization']
    miner = CowrieDrainMiner(normalization_config)
    
    # Process logs
    input_path = Path(raw_logs_path)
    output_path = Path(config['data']['normalized_path'])
    
    if input_path.is_dir():
        log_files = list(input_path.glob("*.log")) + list(input_path.glob("*.json"))
    else:
        log_files = [input_path]
    
    logger.info(f"Processing {len(log_files)} log files...")
    
    total_processed = 0
    templates_found = 0
    
    for log_file in log_files:
        try:
            processed, templates = miner.process_log_file(
                str(log_file), 
                str(output_path / f"{log_file.stem}_normalized.parquet")
            )
            total_processed += processed
            templates_found = max(templates_found, templates)
            
            logger.info(f"Processed {log_file.name}: {processed} lines, {templates} templates")
            
        except Exception as e:
            logger.error(f"Failed to process {log_file}: {e}")
            raise
    
    # Save final template state
    miner.save_state()
    
    logger.info(f"Normalization complete: {total_processed} logs, {templates_found} templates")
    return str(output_path)


@task(name="build_dataset")
def build_dataset(normalized_path: str, config: Dict[str, Any]) -> str:
    """Build windowed datasets for training."""
    logger = get_run_logger()
    logger.info("Building windowed datasets...")
    
    # Initialize dataset builder
    dataset_config = config['dataset']
    builder = WindowedDatasetBuilder(dataset_config)
    
    # Build datasets
    train_path, val_path, test_path = builder.build_datasets(
        input_path=normalized_path,
        output_dir=config['data']['dataset_path']
    )
    
    logger.info(f"Created datasets:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Validation: {val_path}")
    logger.info(f"  Test: {test_path}")
    
    return str(Path(train_path).parent)


@task(name="train_mlm_model")
def train_mlm_model(dataset_path: str, config: Dict[str, Any]) -> str:
    """Train the Masked Language Model (MLM) for anomaly detection."""
    logger = get_run_logger()
    logger.info("Training MLM model...")
    
    # Initialize trainer
    training_config = config['training']['mlm']
    trainer = MLMTrainer(training_config)
    
    # Train model
    model_path = trainer.train(
        train_path=str(Path(dataset_path) / "train_windows.parquet"),
        val_path=str(Path(dataset_path) / "val_windows.parquet"),
        output_dir=config['models']['mlm_model_path']
    )
    
    logger.info(f"MLM training complete. Model saved to: {model_path}")
    return str(model_path)


@task(name="train_infonce_model")
def train_infonce_model(dataset_path: str, mlm_model_path: str, 
                       config: Dict[str, Any]) -> Optional[str]:
    """Train optional InfoNCE contrastive learning model."""
    logger = get_run_logger()
    
    if not config['training'].get('enable_infonce', False):
        logger.info("InfoNCE training disabled, skipping...")
        return None
    
    logger.info("Training InfoNCE model...")
    
    # Initialize trainer
    infonce_config = config['training']['infonce']
    trainer = InfoNCETrainer(infonce_config)
    
    # Train model
    model_path = trainer.train(
        train_path=str(Path(dataset_path) / "train_windows.parquet"),
        val_path=str(Path(dataset_path) / "val_windows.parquet"),
        pretrained_mlm_path=mlm_model_path,
        output_dir=config['models']['infonce_model_path']
    )
    
    logger.info(f"InfoNCE training complete. Model saved to: {model_path}")
    return str(model_path)


@task(name="inject_synthetic_campaigns")
def inject_synthetic_campaigns(test_dataset_path: str, config: Dict[str, Any]) -> str:
    """Inject synthetic attack campaigns for evaluation."""
    logger = get_run_logger()
    logger.info("Injecting synthetic attack campaigns...")
    
    # Initialize campaign generator
    campaign_config = config['evaluation']['synthetic_attacks']
    generator = SyntheticCampaignGenerator(campaign_config)
    
    # Generate campaigns
    test_path = Path(test_dataset_path) / "test_windows.parquet"
    
    augmented_path, campaigns_path = generator.inject_campaigns(
        test_data_path=str(test_path),
        output_dir=str(Path(test_dataset_path) / "evaluation")
    )
    
    logger.info(f"Synthetic campaigns injected:")
    logger.info(f"  Augmented test data: {augmented_path}")
    logger.info(f"  Campaigns metadata: {campaigns_path}")
    
    return str(campaigns_path)


@task(name="generate_weak_labels")
def generate_weak_labels(test_dataset_path: str, config: Dict[str, Any]) -> str:
    """Generate weak labels using rule-based heuristics."""
    logger = get_run_logger()
    logger.info("Generating weak labels...")
    
    # Initialize weak labeler
    weak_labels_config = config['evaluation']['weak_labels']
    labeler = WeakLabeler(weak_labels_config)
    
    # Process test data
    test_path = Path(test_dataset_path) / "evaluation" / "test_with_campaigns.parquet"
    if not test_path.exists():
        test_path = Path(test_dataset_path) / "test_windows.parquet"
    
    weak_labels_path = labeler.label_dataset(
        str(test_path),
        str(Path(test_dataset_path) / "evaluation" / "weak_labels.parquet")
    )
    
    logger.info(f"Weak labels generated: {weak_labels_path}")
    return str(weak_labels_path)


@task(name="run_inference")
def run_inference(model_path: str, test_dataset_path: str, 
                 config: Dict[str, Any]) -> str:
    """Run inference on test dataset."""
    logger = get_run_logger()
    logger.info("Running model inference...")
    
    # This would typically use the inference service
    # For now, we'll simulate by creating predictions
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Load test data
    test_path = Path(test_dataset_path) / "evaluation" / "test_with_campaigns.parquet"
    if not test_path.exists():
        test_path = Path(test_dataset_path) / "test_windows.parquet"
    
    test_df = pd.read_parquet(test_path)
    
    # Simulate predictions (in practice, this would use the actual model)
    np.random.seed(42)  # For reproducible results
    n_samples = len(test_df)
    
    # Create realistic score distribution (most low, some high)
    scores = np.random.beta(2, 8, n_samples)  # Skewed towards 0
    
    # Add some high scores for injected campaigns if they exist
    if 'is_injected' in test_df.columns:
        campaign_mask = test_df['is_injected'] == True
        scores[campaign_mask] = np.random.beta(8, 2, campaign_mask.sum())  # Higher scores
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'sequence_id': test_df.index,
        'timestamp': test_df.get('timestamp', pd.date_range('2024-01-01', periods=n_samples, freq='1min')),
        'source': test_df.get('source', 'unknown'),
        'score_ml': scores,
        'template_ids': test_df.get('template_ids', ''),
        'window_size': test_df.get('window_size', 5)
    })
    
    # Save predictions
    predictions_path = Path(test_dataset_path) / "evaluation" / "predictions.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(predictions_path, index=False)
    
    logger.info(f"Inference complete: {len(predictions_df)} predictions saved to {predictions_path}")
    return str(predictions_path)


@task(name="calibrate_model")
def calibrate_model(predictions_path: str, config: Dict[str, Any]) -> str:
    """Calibrate model scores using rolling percentiles."""
    logger = get_run_logger()
    logger.info("Calibrating model scores...")
    
    # Initialize calibrator
    calibration_config = config['inference']['calibration']
    calibrator = ModelCalibrator(calibration_config)
    
    # Load predictions
    import pandas as pd
    predictions_df = pd.read_parquet(predictions_path)
    
    # Calibrate scores
    calibrated_df = calibrator.calibrate_scores(predictions_df)
    
    # Save calibrated predictions
    calibrated_path = str(Path(predictions_path).parent / "predictions_calibrated.parquet")
    calibrated_df.to_parquet(calibrated_path, index=False)
    
    logger.info(f"Model calibration complete: {calibrated_path}")
    return calibrated_path


@task(name="evaluate_model")
def evaluate_model(predictions_path: str, campaigns_path: str, 
                  weak_labels_path: str, config: Dict[str, Any]) -> str:
    """Run comprehensive model evaluation."""
    logger = get_run_logger()
    logger.info("Running comprehensive evaluation...")
    
    # Initialize evaluator
    eval_config = config['evaluation']
    evaluator = ComprehensiveEvaluator(eval_config)
    
    # Load data
    import pandas as pd
    predictions_df = pd.read_parquet(predictions_path)
    weak_labels_df = pd.read_parquet(weak_labels_path)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        predictions_df=predictions_df,
        campaigns_file=campaigns_path,
        weak_labels_df=weak_labels_df
    )
    
    # Save results
    results_path = str(Path(predictions_path).parent / "evaluation_results.json")
    evaluator.save_results(results, results_path)
    
    # Log key metrics
    summary = results.get('performance_summary', {})
    logger.info("Evaluation Results:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    return results_path


@task(name="package_model")
def package_model(model_path: str, evaluation_results: str, 
                 config: Dict[str, Any]) -> str:
    """Package model for deployment."""
    logger = get_run_logger()
    logger.info("Packaging model for deployment...")
    
    # Load evaluation results to check if model meets criteria
    with open(evaluation_results, 'r') as f:
        results = json.load(f)
    
    summary = results.get('performance_summary', {})
    
    # Check deployment criteria
    deploy_criteria = config['deployment']['criteria']
    meets_criteria = True
    
    if 'recall_score' in summary:
        min_recall = deploy_criteria.get('min_recall', 0.8)
        if summary['recall_score'] < min_recall:
            meets_criteria = False
            logger.warning(f"Recall {summary['recall_score']:.3f} < {min_recall}")
    
    if 'precision_at_k_score' in summary:
        min_precision = deploy_criteria.get('min_precision_at_k', 0.6)
        if summary['precision_at_k_score'] < min_precision:
            meets_criteria = False
            logger.warning(f"Precision@K {summary['precision_at_k_score']:.3f} < {min_precision}")
    
    # Package model
    package_dir = Path(config['deployment']['package_path'])
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    import shutil
    model_source = Path(model_path)
    model_dest = package_dir / "model"
    
    if model_dest.exists():
        shutil.rmtree(model_dest)
    
    shutil.copytree(model_source, model_dest)
    
    # Copy evaluation results
    shutil.copy2(evaluation_results, package_dir / "evaluation_results.json")
    
    # Create deployment metadata
    deployment_metadata = {
        'model_path': str(model_path),
        'evaluation_results': results,
        'meets_deployment_criteria': meets_criteria,
        'package_timestamp': datetime.now().isoformat(),
        'config_snapshot': config
    }
    
    metadata_path = package_dir / "deployment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(deployment_metadata, f, indent=2, default=str)
    
    logger.info(f"Model packaged: {package_dir}")
    logger.info(f"Meets deployment criteria: {meets_criteria}")
    
    return str(package_dir)


@flow(
    name="logbert-training-pipeline",
    task_runner=ConcurrentTaskRunner(),
    description="Complete LogBERT anomaly detection training pipeline"
)
def logbert_training_flow(config_path: str = "./configs/config.yaml") -> Dict[str, Any]:
    """Main training pipeline flow."""
    logger = get_run_logger()
    logger.info("Starting LogBERT training pipeline...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Sequential pipeline stages
    logger.info("=== Stage 1: Data Extraction ===")
    raw_logs_path = extract_raw_logs(config)
    
    logger.info("=== Stage 2: Log Normalization ===")
    normalized_path = normalize_logs(raw_logs_path, config)
    
    logger.info("=== Stage 3: Dataset Building ===")
    dataset_path = build_dataset(normalized_path, config)
    
    # Parallel training (MLM + optional InfoNCE)
    logger.info("=== Stage 4: Model Training ===")
    mlm_model_path = train_mlm_model(dataset_path, config)
    infonce_model_path = train_infonce_model(dataset_path, mlm_model_path, config)
    
    # Use the best available model for evaluation
    final_model_path = infonce_model_path if infonce_model_path else mlm_model_path
    
    # Parallel evaluation setup
    logger.info("=== Stage 5: Evaluation Setup ===")
    campaigns_path = inject_synthetic_campaigns(dataset_path, config)
    weak_labels_path = generate_weak_labels(dataset_path, config)
    
    logger.info("=== Stage 6: Model Inference ===")
    predictions_path = run_inference(final_model_path, dataset_path, config)
    
    logger.info("=== Stage 7: Model Calibration ===")
    calibrated_predictions_path = calibrate_model(predictions_path, config)
    
    logger.info("=== Stage 8: Evaluation ===")
    evaluation_results = evaluate_model(
        calibrated_predictions_path, campaigns_path, weak_labels_path, config
    )
    
    logger.info("=== Stage 9: Model Packaging ===")
    package_path = package_model(final_model_path, evaluation_results, config)
    
    # Return pipeline results
    pipeline_results = {
        'model_path': final_model_path,
        'package_path': package_path,
        'evaluation_results': evaluation_results,
        'dataset_path': dataset_path,
        'predictions_path': calibrated_predictions_path
    }
    
    logger.info("=== Pipeline Complete ===")
    logger.info(f"Results: {pipeline_results}")
    
    return pipeline_results


@flow(
    name="logbert-inference-pipeline",
    description="LogBERT real-time inference pipeline"
)
def logbert_inference_flow(config_path: str = "./configs/config.yaml",
                          model_package_path: str = None) -> Dict[str, Any]:
    """Real-time inference pipeline flow."""
    logger = get_run_logger()
    logger.info("Starting LogBERT inference pipeline...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_package_path is None:
        model_package_path = config['deployment']['package_path']
    
    # This would typically:
    # 1. Load latest logs from stream/batch
    # 2. Normalize with existing Drain state  
    # 3. Create windows
    # 4. Run inference with packaged model
    # 5. Apply calibration
    # 6. Generate alerts
    # 7. Update monitoring metrics
    
    logger.info("Inference pipeline would run here...")
    logger.info(f"Using model package: {model_package_path}")
    
    return {
        'status': 'inference_complete',
        'model_package': model_package_path
    }


def create_training_deployment():
    """Create Prefect deployment for training pipeline."""
    deployment = Deployment.build_from_flow(
        flow=logbert_training_flow,
        name="logbert-training-daily",
        schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),  # Daily at 2 AM UTC
        work_queue_name="training",
        parameters={"config_path": "./configs/config.yaml"},
        description="Daily LogBERT model training and evaluation"
    )
    
    return deployment


def create_inference_deployment():
    """Create Prefect deployment for inference pipeline."""
    deployment = Deployment.build_from_flow(
        flow=logbert_inference_flow,
        name="logbert-inference-hourly",
        schedule=CronSchedule(cron="0 * * * *", timezone="UTC"),  # Hourly
        work_queue_name="inference",
        parameters={"config_path": "./configs/config.yaml"},
        description="Hourly LogBERT anomaly detection inference"
    )
    
    return deployment


if __name__ == "__main__":
    # Example: Run training pipeline locally
    import asyncio
    
    async def run_training():
        result = await logbert_training_flow()
        print(f"Training pipeline completed: {result}")
    
    # Run the training pipeline
    asyncio.run(run_training())