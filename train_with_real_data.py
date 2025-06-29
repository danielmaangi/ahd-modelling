#!/usr/bin/env python3
"""
Simple script to train the HIV disease prediction model using real data.
This demonstrates the complete pipeline from data loading to model training.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import HIVDataLoader
from models.train_model import HIVModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train HIV disease prediction model using real data."""
    
    logger.info("Starting HIV Disease Prediction Model Training with Real Data")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load and process real data
        logger.info("Step 1: Loading and processing real HIV patient data...")
        loader = HIVDataLoader()
        
        # Use a sample for demonstration (10,000 records)
        # Remove sample_size=10000 to use all data
        df_processed = loader.load_and_process_data(
            "data/raw/PLHIV Linelist.csv", 
            sample_size=10000
        )
        
        logger.info(f"Processed {len(df_processed)} patient records")
        logger.info(f"Advanced HIV prevalence: {df_processed['advanced_hiv'].mean():.1%}")
        
        # Step 2: Train the model
        logger.info("Step 2: Training XGBoost model...")
        trainer = HIVModelTrainer()
        
        # Prepare data for training
        X, y, feature_columns = trainer.prepare_data(df_processed)
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Features: {len(feature_columns)}")
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        # Train model
        model = trainer.train_model(X_train_scaled, y_train)
        
        # Step 3: Evaluate the model
        logger.info("Step 3: Evaluating model performance...")
        
        # Cross-validation
        cv_results = trainer.cross_validate(X_train_scaled, y_train)
        logger.info("Cross-validation results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if f'{metric}_mean' in cv_results:
                mean_score = cv_results[f'{metric}_mean']
                std_score = cv_results[f'{metric}_std']
                logger.info(f"  {metric}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        # Test set evaluation
        evaluation_results = trainer.evaluate_model(X_test_scaled, y_test)
        logger.info("Test set evaluation:")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"  F1-Score: {evaluation_results['f1']:.4f}")
        logger.info(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        
        # Step 4: Save the model
        logger.info("Step 4: Saving trained model...")
        model_dir = "models/real_data_model"
        trainer.save_model(model_dir)
        logger.info(f"Model saved to {model_dir}")
        
        # Feature importance
        if 'feature_importance' in evaluation_results:
            logger.info("Top 10 most important features:")
            feature_importance = evaluation_results['feature_importance']
            for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        logger.info("=" * 60)
        logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features used: {len(feature_columns)}")
        logger.info(f"  Final test accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Final test ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        logger.info(f"  Model saved to: {model_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
