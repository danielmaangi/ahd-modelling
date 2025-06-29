"""
HIV Disease Prediction - Model Training Module

This module handles the training of XGBoost models for HIV disease prediction.
It includes hyperparameter tuning, cross-validation, model evaluation, and
model persistence functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import yaml
import json
import joblib
from datetime import datetime

# ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)


class HIVModelTrainer:
    """
    Comprehensive model trainer for HIV disease prediction using XGBoost.
    
    Features:
    - Automated hyperparameter tuning
    - Cross-validation with multiple metrics
    - Feature importance analysis
    - Model evaluation and visualization
    - Model persistence and metadata tracking
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.training_history = {}
        self.evaluation_results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load training configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        
        # Default configuration
        return {
            'model': {
                'parameters': {
                    'objective': 'binary:logistic',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'eval_metric': 'auc'
                }
            },
            'training': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'stratify': True,
                'cv_folds': 5
            },
            'evaluation': {
                'primary_metric': 'roc_auc',
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'advanced_hiv') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y, feature_columns)
        """
        logger.info("Preparing data for training...")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Separate features and target
        y = df[target_col].values
        
        # Get feature columns (exclude target and non-predictive columns)
        exclude_cols = [
            target_col, 'patient_id', 'data_source', 'generation_date',
            'age_group', 'cd4_category', 'viral_load_category', 'bmi_category',
            'adherence_category', 'time_category'  # Exclude categorical versions
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical columns that need encoding
        df_processed = df.copy()
        
        # Encode gender if present
        if 'gender' in df_processed.columns and 'gender_encoded' not in df_processed.columns:
            self.label_encoder = LabelEncoder()
            df_processed['gender_encoded'] = self.label_encoder.fit_transform(df_processed['gender'])
            feature_cols.append('gender_encoded')
            if 'gender' in feature_cols:
                feature_cols.remove('gender')
        
        # One-hot encode comorbidities if not already done
        if 'comorbidities' in df_processed.columns:
            if not any(col.startswith('comorbidity_') for col in df_processed.columns):
                comorbidity_dummies = pd.get_dummies(df_processed['comorbidities'], prefix='comorbidity')
                df_processed = pd.concat([df_processed, comorbidity_dummies], axis=1)
                feature_cols.extend(comorbidity_dummies.columns.tolist())
            
            if 'comorbidities' in feature_cols:
                feature_cols.remove('comorbidities')
        
        # Select only numerical features for training
        numerical_features = []
        for col in feature_cols:
            if col in df_processed.columns and df_processed[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
        
        self.feature_columns = numerical_features
        X = df_processed[self.feature_columns].values
        
        # Handle any remaining NaN values
        if np.isnan(X).any():
            logger.warning("Found NaN values in features, filling with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y, self.feature_columns
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.config['training']['test_size']
        random_state = self.config['training']['random_state']
        stratify = y if self.config['training']['stratify'] else None
        
        logger.info(f"Splitting data with test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Training target distribution: {np.bincount(y_train)}")
        logger.info(f"Test target distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info("Scaling features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> xgb.XGBClassifier:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Get model parameters from config
        model_params = self.config['model']['parameters'].copy()
        
        # Initialize model
        self.model = xgb.XGBClassifier(**model_params)
        
        # Prepare evaluation set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train model
        start_time = datetime.now()
        
        if eval_set:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store training history
        self.training_history = {
            'training_time_seconds': training_time,
            'n_estimators': self.model.n_estimators,
            'feature_importances': self.model.feature_importances_.tolist(),
            'training_samples': len(X_train),
            'training_date': datetime.now().isoformat()
        }
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        return self.model
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            method: str = 'grid_search') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: Tuning method ('grid_search' or 'random_search')
            
        Returns:
            Dictionary with best parameters and scores
        """
        logger.info(f"Starting hyperparameter tuning using {method}...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='auc'
        )
        
        # Choose search method
        cv_folds = self.config['training']['cv_folds']
        
        if method == 'grid_search':
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv_folds, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
        elif method == 'random_search':
            search = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=50, cv=cv_folds, scoring='roc_auc',
                n_jobs=-1, verbose=1, random_state=42
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
        
        # Perform search
        start_time = datetime.now()
        search.fit(X_train, y_train)
        tuning_time = (datetime.now() - start_time).total_seconds()
        
        # Update model with best parameters
        self.model = search.best_estimator_
        
        # Store results
        tuning_results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'tuning_method': method,
            'tuning_time_seconds': tuning_time,
            'cv_folds': cv_folds
        }
        
        logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        logger.info(f"Best score: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")
        
        return tuning_results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform cross-validation evaluation.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with cross-validation scores
        """
        logger.info("Performing cross-validation...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        cv_folds = self.config['training']['cv_folds']
        metrics = self.config['evaluation']['metrics']
        
        cv_results = {}
        
        for metric in metrics:
            if metric == 'roc_auc':
                scoring = 'roc_auc'
            elif metric == 'accuracy':
                scoring = 'accuracy'
            elif metric == 'precision':
                scoring = 'precision'
            elif metric == 'recall':
                scoring = 'recall'
            elif metric == 'f1':
                scoring = 'f1'
            else:
                continue
            
            scores = cross_val_score(
                self.model, X, y, cv=cv_folds, scoring=scoring
            )
            
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
            cv_results[f'{metric}_scores'] = scores.tolist()
        
        logger.info("Cross-validation completed")
        for metric in metrics:
            if f'{metric}_mean' in cv_results:
                mean_score = cv_results[f'{metric}_mean']
                std_score = cv_results[f'{metric}_std']
                logger.info(f"{metric}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        return cv_results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model on test data...")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        evaluation_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        evaluation_results['classification_report'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        evaluation_results['confusion_matrix'] = cm.tolist()
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        evaluation_results['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        }
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        evaluation_results['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            evaluation_results['feature_importance'] = feature_importance
        
        self.evaluation_results = evaluation_results
        
        # Log key metrics
        logger.info("Model evaluation completed:")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"  F1-Score: {evaluation_results['f1']:.4f}")
        logger.info(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        
        return evaluation_results
    
    def plot_evaluation_results(self, save_path: Optional[str] = None) -> None:
        """
        Create evaluation plots.
        
        Args:
            save_path: Directory to save plots (optional)
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available. Run evaluate_model() first.")
            return
        
        logger.info("Creating evaluation plots...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HIV Disease Prediction Model Evaluation', fontsize=16)
        
        # 1. Confusion Matrix
        cm = np.array(self.evaluation_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        roc_data = self.evaluation_results['roc_curve']
        auc_score = self.evaluation_results['roc_auc']
        axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'], 
                       label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall Curve
        pr_data = self.evaluation_results['pr_curve']
        ap_score = self.evaluation_results['average_precision']
        axes[1, 0].plot(pr_data['recall'], pr_data['precision'],
                       label=f'PR Curve (AP = {ap_score:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Feature Importance (Top 10)
        if 'feature_importance' in self.evaluation_results:
            feature_imp = self.evaluation_results['feature_importance']
            top_features = list(feature_imp.items())[:10]
            features, importances = zip(*top_features)
            
            y_pos = np.arange(len(features))
            axes[1, 1].barh(y_pos, importances)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(features)
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plot_file = Path(save_path) / 'model_evaluation.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {plot_file}")
        
        plt.show()
    
    def save_model(self, model_dir: str) -> None:
        """
        Save the trained model and associated artifacts.
        
        Args:
            model_dir: Directory to save model artifacts
        """
        logger.info(f"Saving model to {model_dir}...")
        
        # Create directory
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = Path(model_dir) / 'hiv_xgboost_model.pkl'
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = Path(model_dir) / 'scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save label encoder
        if self.label_encoder is not None:
            encoder_path = Path(model_dir) / 'label_encoder_gender.pkl'
            joblib.dump(self.label_encoder, encoder_path)
            logger.info(f"Label encoder saved to {encoder_path}")
        
        # Save feature columns
        if self.feature_columns is not None:
            features_path = Path(model_dir) / 'feature_columns.pkl'
            joblib.dump(self.feature_columns, features_path)
            logger.info(f"Feature columns saved to {features_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'model_version': '1.0.0',
            'training_date': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'training_history': self.training_history,
            'evaluation_results': {
                k: v for k, v in self.evaluation_results.items()
                if k not in ['roc_curve', 'pr_curve']  # Exclude large arrays
            },
            'config': self.config
        }
        
        metadata_path = Path(model_dir) / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def load_model(self, model_dir: str) -> None:
        """
        Load a trained model and associated artifacts.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        logger.info(f"Loading model from {model_dir}...")
        
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / 'hiv_xgboost_model.pkl'
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        
        # Load scaler
        scaler_path = model_dir / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load label encoder
        encoder_path = model_dir / 'label_encoder_gender.pkl'
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"Label encoder loaded from {encoder_path}")
        
        # Load feature columns
        features_path = model_dir / 'feature_columns.pkl'
        if features_path.exists():
            self.feature_columns = joblib.load(features_path)
            logger.info(f"Feature columns loaded from {features_path}")
        
        # Load metadata
        metadata_path = model_dir / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.training_history = metadata.get('training_history', {})
            self.evaluation_results = metadata.get('evaluation_results', {})
            logger.info(f"Model metadata loaded from {metadata_path}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HIV disease prediction model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data CSV file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='models/',
                       help='Output directory for model artifacts')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--plots', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    logger.info(f"Loading training data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Initialize trainer
    trainer = HIVModelTrainer(config_path=args.config)
    
    # Prepare data
    X, y, feature_columns = trainer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    # Hyperparameter tuning if requested
    if args.tune:
        tuning_results = trainer.hyperparameter_tuning(X_train_scaled, y_train)
        logger.info(f"Best parameters: {tuning_results['best_params']}")
    else:
        # Train model with default parameters
        trainer.train_model(X_train_scaled, y_train)
    
    # Cross-validation
    cv_results = trainer.cross_validate(X_train_scaled, y_train)
    
    # Evaluate on test set
    evaluation_results = trainer.evaluate_model(X_test_scaled, y_test)
    
    # Generate plots if requested
    if args.plots:
        trainer.plot_evaluation_results(save_path=args.output)
    
    # Save model
    trainer.save_model(args.output)
    
    # Print summary
    print(f"\nModel Training Complete!")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Test ROC-AUC: {evaluation_results['roc_auc']:.4f}")
    print(f"Test F1-Score: {evaluation_results['f1']:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
