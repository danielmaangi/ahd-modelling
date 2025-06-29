#!/usr/bin/env python3
"""
HIV Disease Prediction - Complete Training Pipeline

This script runs the complete training pipeline for the HIV disease prediction model,
including data generation, preprocessing, feature engineering, model training,
evaluation, and deployment preparation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import our modules
from data.data_generator import HIVDataGenerator
from data.data_preprocessing import HIVDataPreprocessor
from data.feature_engineering import HIVFeatureEngineer
from models.train_model import HIVModelTrainer
from models.evaluate_model import HIVModelEvaluator
from utils.config import load_config
from utils.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def setup_directories(base_dir: Path) -> None:
    """Create necessary directories for the pipeline."""
    directories = [
        'data/raw',
        'data/processed',
        'data/synthetic',
        'models',
        'logs',
        'reports',
        'plots'
    ]
    
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Created necessary directories")


def generate_synthetic_data(config: dict, output_dir: Path) -> Path:
    """Generate synthetic HIV dataset."""
    logger.info("Starting synthetic data generation...")
    
    # Initialize data generator
    generator = HIVDataGenerator(config_path=None)  # Use default config for now
    
    # Generate dataset
    n_samples = config.get('data_generation', {}).get('n_samples', 1000)
    dataset = generator.generate_dataset(n_samples=n_samples)
    
    # Save dataset
    output_file = output_dir / 'synthetic_hiv_data.csv'
    generator.save_dataset(dataset, str(output_file), include_summary=True)
    
    logger.info(f"Generated {len(dataset)} synthetic samples")
    logger.info(f"Advanced HIV prevalence: {dataset['advanced_hiv'].mean():.1%}")
    
    return output_file


def preprocess_data(data_file: Path, config: dict, output_dir: Path) -> tuple:
    """Preprocess the dataset."""
    logger.info("Starting data preprocessing...")
    
    # Load data
    import pandas as pd
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples from {data_file}")
    
    # Initialize preprocessor
    preprocessor = HIVDataPreprocessor(config_path=None)  # Use default config
    
    # Run preprocessing pipeline
    df_processed, feature_columns = preprocessor.preprocess_pipeline(df, fit=True)
    
    # Save processed data
    processed_file = output_dir / 'processed_hiv_data.csv'
    df_processed.to_csv(processed_file, index=False)
    
    # Save preprocessor
    preprocessor_file = output_dir / 'preprocessor.pkl'
    preprocessor.save_preprocessors(str(preprocessor_file))
    
    logger.info(f"Preprocessing complete. Output shape: {df_processed.shape}")
    logger.info(f"Feature columns: {len(feature_columns)}")
    
    return processed_file, feature_columns


def engineer_features(data_file: Path, output_dir: Path) -> Path:
    """Engineer additional features."""
    logger.info("Starting feature engineering...")
    
    # Load processed data
    import pandas as pd
    df = pd.read_csv(data_file)
    
    # Initialize feature engineer
    engineer = HIVFeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.engineer_all_features(df)
    
    # Feature selection (optional)
    selected_features = engineer.select_best_features(df_engineered, k=20)
    
    # Keep selected features plus target and ID columns
    keep_cols = selected_features + ['advanced_hiv']
    if 'patient_id' in df_engineered.columns:
        keep_cols.append('patient_id')
    
    df_final = df_engineered[keep_cols]
    
    # Save engineered data
    engineered_file = output_dir / 'engineered_hiv_data.csv'
    df_final.to_csv(engineered_file, index=False)
    
    # Save feature engineering summary
    summary = engineer.get_feature_summary()
    import json
    summary_file = output_dir / 'feature_engineering_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Feature engineering complete. Final shape: {df_final.shape}")
    logger.info(f"Selected {len(selected_features)} features")
    
    return engineered_file


def train_model(data_file: Path, config: dict, output_dir: Path) -> Path:
    """Train the XGBoost model."""
    logger.info("Starting model training...")
    
    # Load data
    import pandas as pd
    df = pd.read_csv(data_file)
    
    # Initialize trainer
    trainer = HIVModelTrainer(config_path=None)  # Use default config
    
    # Prepare data
    X, y, feature_columns = trainer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    # Train model
    model = trainer.train_model(X_train_scaled, y_train)
    
    # Cross-validation
    cv_results = trainer.cross_validate(X_train_scaled, y_train)
    logger.info("Cross-validation results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        if f'{metric}_mean' in cv_results:
            mean_score = cv_results[f'{metric}_mean']
            std_score = cv_results[f'{metric}_std']
            logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
    
    # Evaluate on test set
    evaluation_results = trainer.evaluate_model(X_test_scaled, y_test)
    logger.info("Test set evaluation:")
    logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
    logger.info(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
    logger.info(f"  F1-Score: {evaluation_results['f1']:.4f}")
    
    # Save model
    model_dir = output_dir / 'model'
    trainer.save_model(str(model_dir))
    
    # Generate plots
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    trainer.plot_evaluation_results(save_path=str(plots_dir))
    
    logger.info(f"Model training complete. Saved to {model_dir}")
    
    return model_dir


def evaluate_model(model_dir: Path, test_data_file: Path, output_dir: Path) -> None:
    """Comprehensive model evaluation."""
    logger.info("Starting comprehensive model evaluation...")
    
    # Initialize evaluator
    evaluator = HIVModelEvaluator()
    
    # Load model artifacts
    artifacts = evaluator.load_model_artifacts(str(model_dir))
    
    # Load test data
    import pandas as pd
    test_df = pd.read_csv(test_data_file)
    
    # Prepare test data
    X_test = test_df[artifacts['feature_columns']].values
    y_test = test_df['advanced_hiv'].values
    
    # Scale features if scaler is available
    if 'scaler' in artifacts:
        X_test = artifacts['scaler'].transform(X_test)
    
    # Make predictions
    model = artifacts['model']
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
    
    # Evaluate calibration
    calibration_results = evaluator.evaluate_calibration(y_test, y_pred_proba)
    
    # Cross-validation
    cv_results = evaluator.cross_validate_model(model, X_test, y_test)
    
    # Combine results
    evaluation_results = {
        'metrics': metrics,
        'calibration': calibration_results,
        'cross_validation': cv_results,
        'confusion_matrix': [[metrics['true_negatives'], metrics['false_positives']],
                           [metrics['false_negatives'], metrics['true_positives']]]
    }
    
    # Generate comprehensive report
    report_file = output_dir / 'evaluation_report.md'
    report = evaluator.create_evaluation_report(
        'HIV_XGBoost_Model', 
        evaluation_results,
        save_path=str(report_file)
    )
    
    # Create interactive plots
    plots_dir = output_dir / 'evaluation_plots'
    plots_dir.mkdir(exist_ok=True)
    evaluator.create_interactive_plots(
        y_test, y_pred, y_pred_proba,
        save_path=str(plots_dir)
    )
    
    # Save evaluation results
    results_file = output_dir / 'evaluation_results.json'
    import json
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info("Comprehensive evaluation complete")
    logger.info(f"Final model performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")


def create_deployment_package(model_dir: Path, output_dir: Path) -> None:
    """Create deployment package."""
    logger.info("Creating deployment package...")
    
    deployment_dir = output_dir / 'deployment'
    deployment_dir.mkdir(exist_ok=True)
    
    # Copy model artifacts
    import shutil
    model_deployment_dir = deployment_dir / 'models'
    if model_deployment_dir.exists():
        shutil.rmtree(model_deployment_dir)
    shutil.copytree(model_dir, model_deployment_dir)
    
    # Create deployment script
    deployment_script = deployment_dir / 'deploy.py'
    with open(deployment_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
HIV Disease Prediction - Deployment Script

This script deploys the trained HIV disease prediction model.
\"\"\"

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from api.app import main

if __name__ == "__main__":
    # Set environment variables
    import os
    os.environ['MODEL_PATH'] = str(Path(__file__).parent / 'models')
    os.environ['CONFIG_PATH'] = str(Path(__file__).parent.parent / 'config' / 'app_config.yaml')
    
    # Run the API
    main()
""")
    
    # Make deployment script executable
    deployment_script.chmod(0o755)
    
    # Create requirements file for deployment
    deployment_requirements = deployment_dir / 'requirements.txt'
    with open(deployment_requirements, 'w') as f:
        f.write("""fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.2
joblib==1.3.2
PyYAML==6.0.1
prometheus-fastapi-instrumentator==6.1.0
""")
    
    # Create Docker file
    dockerfile = deployment_dir / 'Dockerfile'
    with open(dockerfile, 'w') as f:
        f.write("""FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \\
    gcc \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && curl -LsSf https://astral.sh/uv/install.sh | sh \\
    && mv /root/.cargo/bin/uv /usr/local/bin/

# Copy requirements and install Python dependencies with uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "deploy.py", "--host", "0.0.0.0", "--port", "5000"]
""")
    
    logger.info(f"Deployment package created in {deployment_dir}")


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='HIV Disease Prediction Training Pipeline')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for all artifacts')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--skip-data-generation', action='store_true',
                       help='Skip synthetic data generation')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Use existing data file instead of generating')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip comprehensive evaluation')
    parser.add_argument('--create-deployment', action='store_true',
                       help='Create deployment package')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Setup directories
    output_dir = Path(args.output_dir)
    setup_directories(output_dir)
    
    # Load configuration
    config = load_config(args.config) if Path(args.config).exists() else {}
    
    logger.info("Starting HIV Disease Prediction Training Pipeline")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Step 1: Data Generation or Loading
        if args.skip_data_generation and args.data_file:
            data_file = Path(args.data_file)
            logger.info(f"Using existing data file: {data_file}")
        elif not args.skip_data_generation:
            # Update config with command line arguments
            if 'data_generation' not in config:
                config['data_generation'] = {}
            config['data_generation']['n_samples'] = args.n_samples
            
            data_file = generate_synthetic_data(config, output_dir / 'data' / 'synthetic')
        else:
            raise ValueError("Must provide --data-file when using --skip-data-generation")
        
        # Step 2: Data Preprocessing
        processed_file, feature_columns = preprocess_data(
            data_file, config, output_dir / 'data' / 'processed'
        )
        
        # Step 3: Feature Engineering
        engineered_file = engineer_features(
            processed_file, output_dir / 'data' / 'processed'
        )
        
        # Step 4: Model Training
        model_dir = train_model(
            engineered_file, config, output_dir
        )
        
        # Step 5: Comprehensive Evaluation
        if not args.skip_evaluation:
            evaluate_model(
                model_dir, engineered_file, output_dir / 'evaluation'
            )
        
        # Step 6: Create Deployment Package
        if args.create_deployment:
            create_deployment_package(model_dir, output_dir)
        
        # Pipeline Summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Data file: {data_file}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        if not args.skip_evaluation:
            # Load final metrics
            results_file = output_dir / 'evaluation' / 'evaluation_results.json'
            if results_file.exists():
                import json
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                metrics = results.get('metrics', {})
                logger.info("Final Model Performance:")
                logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                logger.info(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                logger.info(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
                logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
                logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
