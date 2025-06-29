"""
HIV Disease Prediction - Model Utilities Module

This module provides utility functions for model operations including
model loading, prediction preprocessing, batch processing, and model
management utilities for the HIV disease prediction system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import json
import joblib
from datetime import datetime
import warnings

# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

# Configure logging
logger = logging.getLogger(__name__)


class HIVModelPredictor:
    """
    Production-ready model predictor for HIV disease prediction.
    
    Handles model loading, input preprocessing, prediction generation,
    and output formatting for both single and batch predictions.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the model predictor.
        
        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.metadata = None
        self.is_loaded = False
        
        # Load model artifacts
        self.load_model_artifacts()
    
    def load_model_artifacts(self) -> None:
        """Load all model artifacts from the model directory."""
        logger.info(f"Loading model artifacts from {self.model_dir}")
        
        try:
            # Load model
            model_path = self.model_dir / 'hiv_xgboost_model.pkl'
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load scaler
            scaler_path = self.model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            
            # Load label encoder
            encoder_path = self.model_dir / 'label_encoder_gender.pkl'
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info("Label encoder loaded successfully")
            
            # Load feature columns
            features_path = self.model_dir / 'feature_columns.pkl'
            if features_path.exists():
                self.feature_columns = joblib.load(features_path)
                logger.info("Feature columns loaded successfully")
            else:
                raise FileNotFoundError(f"Feature columns file not found: {features_path}")
            
            # Load metadata
            metadata_path = self.model_dir / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("Model metadata loaded successfully")
            
            self.is_loaded = True
            logger.info("All model artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}")
            raise
    
    def preprocess_input(self, data: Union[Dict[str, Any], pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data as dictionary or DataFrame
            
        Returns:
            Preprocessed feature array
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts not loaded")
        
        # Convert to DataFrame if dictionary
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Apply feature engineering (same as training)
        df = self._apply_feature_engineering(df)
        
        # Encode categorical features
        df = self._encode_categorical_features(df)
        
        # Select feature columns
        try:
            X = df[self.feature_columns].values
        except KeyError as e:
            missing_cols = set(self.feature_columns) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found NaN values in input, filling with median")
            # Use simple median imputation for missing values
            for i in range(X.shape[1]):
                col_data = X[:, i]
                if np.isnan(col_data).any():
                    median_val = np.nanmedian(col_data)
                    X[np.isnan(col_data), i] = median_val
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        df_processed = df.copy()
        
        # CD4 to viral load ratio
        if 'cd4_count' in df.columns and 'viral_load' in df.columns:
            df_processed['cd4_viral_ratio'] = (
                df_processed['cd4_count'] / (df_processed['viral_load'] + 1)
            )
        
        # Log-transformed viral load
        if 'viral_load' in df.columns:
            df_processed['log_viral_load'] = np.log10(df_processed['viral_load'] + 1)
        
        # Binary indicators
        if 'cd4_count' in df.columns:
            df_processed['low_cd4'] = (df_processed['cd4_count'] < 350).astype(int)
        
        if 'viral_load' in df.columns:
            df_processed['high_viral_load'] = (df_processed['viral_load'] > 1000).astype(int)
        
        if 'art_adherence' in df.columns:
            df_processed['poor_adherence'] = (df_processed['art_adherence'] < 0.8).astype(int)
        
        return df_processed
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df_processed = df.copy()
        
        # Encode gender
        if 'gender' in df.columns and self.label_encoder is not None:
            # Handle unseen categories
            try:
                df_processed['gender_encoded'] = self.label_encoder.transform(df_processed['gender'])
            except ValueError:
                # Handle unseen gender values by mapping to most common
                logger.warning("Unseen gender values found, using default encoding")
                df_processed['gender_encoded'] = 0  # Default to first class
        
        # One-hot encode comorbidities
        if 'comorbidities' in df.columns:
            comorbidity_options = ['None', 'Diabetes', 'Hypertension', 'Both']
            for option in comorbidity_options:
                col_name = f'comorbidity_{option}'
                df_processed[col_name] = (df_processed['comorbidities'] == option).astype(int)
        
        return df_processed
    
    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts not loaded")
        
        # Preprocess input
        X = self.preprocess_input(data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        prediction_proba = self.model.predict_proba(X)[0]
        
        # Calculate confidence and risk level
        confidence = abs(prediction_proba[1] - 0.5) * 2
        
        if prediction_proba[1] > 0.7:
            risk_level = "High"
        elif prediction_proba[1] > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Format result
        result = {
            'prediction': int(prediction),
            'probability': {
                'no_advanced_hiv': float(prediction_proba[0]),
                'advanced_hiv': float(prediction_proba[1])
            },
            'risk_level': risk_level,
            'confidence': float(confidence),
            'model_version': self.metadata.get('model_version', '1.0.0') if self.metadata else '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, data: List[Dict[str, Any]], 
                     max_batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            data: List of input data dictionaries
            max_batch_size: Maximum batch size to process
            
        Returns:
            List of prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts not loaded")
        
        if len(data) > max_batch_size:
            raise ValueError(f"Batch size {len(data)} exceeds maximum {max_batch_size}")
        
        results = []
        
        try:
            # Convert to DataFrame for batch processing
            df = pd.DataFrame(data)
            
            # Preprocess batch
            X = self.preprocess_input(df)
            
            # Make batch predictions
            predictions = self.model.predict(X)
            predictions_proba = self.model.predict_proba(X)
            
            # Process each result
            for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
                confidence = abs(proba[1] - 0.5) * 2
                
                if proba[1] > 0.7:
                    risk_level = "High"
                elif proba[1] > 0.3:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                result = {
                    'prediction': int(pred),
                    'probability': {
                        'no_advanced_hiv': float(proba[0]),
                        'advanced_hiv': float(proba[1])
                    },
                    'risk_level': risk_level,
                    'confidence': float(confidence),
                    'model_version': self.metadata.get('model_version', '1.0.0') if self.metadata else '1.0.0',
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
        
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return error for each input
            for _ in data:
                results.append({'error': str(e)})
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            raise RuntimeError("Model artifacts not loaded")
        
        info = {
            'model_type': 'XGBoost',
            'model_loaded': True,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'has_scaler': self.scaler is not None,
            'has_label_encoder': self.label_encoder is not None,
            'model_dir': str(self.model_dir)
        }
        
        # Add metadata if available
        if self.metadata:
            info.update(self.metadata)
        
        return info


class ModelValidator:
    """
    Utility class for validating model inputs and outputs.
    """
    
    @staticmethod
    def validate_patient_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate patient data for prediction.
        
        Args:
            data: Patient data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Required fields
        required_fields = [
            'age', 'viral_load', 'time_since_diagnosis',
            'art_adherence', 'bmi', 'hemoglobin', 'albumin', 'gender', 'comorbidities'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None:
                errors.append(f"Field {field} cannot be None")
        
        if errors:
            return False, errors
        
        # Validate ranges
        validations = [
            ('age', data.get('age'), 0, 120),
            ('viral_load', data.get('viral_load'), 0, 10000000),
            ('time_since_diagnosis', data.get('time_since_diagnosis'), 0, 50),
            ('art_adherence', data.get('art_adherence'), 0, 1),
            ('bmi', data.get('bmi'), 10, 50),
            ('hemoglobin', data.get('hemoglobin'), 5, 20),
            ('albumin', data.get('albumin'), 1, 6)
        ]
        
        for field, value, min_val, max_val in validations:
            if value is not None:
                try:
                    float_val = float(value)
                    if not (min_val <= float_val <= max_val):
                        errors.append(f"{field} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Validate categorical fields
        if 'gender' in data:
            valid_genders = ['M', 'F', 'Male', 'Female']
            if data['gender'] not in valid_genders:
                errors.append(f"gender must be one of: {valid_genders}")
        
        if 'comorbidities' in data:
            valid_comorbidities = ['None', 'Diabetes', 'Hypertension', 'Both']
            if data['comorbidities'] not in valid_comorbidities:
                errors.append(f"comorbidities must be one of: {valid_comorbidities}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_prediction_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate prediction result format.
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Required fields in result
        required_fields = ['prediction', 'probability', 'risk_level', 'confidence']
        
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field in result: {field}")
        
        # Validate prediction
        if 'prediction' in result:
            if result['prediction'] not in [0, 1]:
                errors.append("prediction must be 0 or 1")
        
        # Validate probability
        if 'probability' in result:
            prob = result['probability']
            if not isinstance(prob, dict):
                errors.append("probability must be a dictionary")
            else:
                required_prob_keys = ['no_advanced_hiv', 'advanced_hiv']
                for key in required_prob_keys:
                    if key not in prob:
                        errors.append(f"Missing probability key: {key}")
                    else:
                        try:
                            val = float(prob[key])
                            if not (0 <= val <= 1):
                                errors.append(f"Probability {key} must be between 0 and 1")
                        except (ValueError, TypeError):
                            errors.append(f"Probability {key} must be a number")
        
        # Validate risk level
        if 'risk_level' in result:
            valid_risk_levels = ['Low', 'Medium', 'High']
            if result['risk_level'] not in valid_risk_levels:
                errors.append(f"risk_level must be one of: {valid_risk_levels}")
        
        # Validate confidence
        if 'confidence' in result:
            try:
                conf = float(result['confidence'])
                if not (0 <= conf <= 1):
                    errors.append("confidence must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("confidence must be a number")
        
        return len(errors) == 0, errors


class ModelPerformanceTracker:
    """
    Utility class for tracking model performance in production.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize performance tracker.
        
        Args:
            log_file: Path to log file for storing performance data
        """
        self.log_file = log_file
        self.predictions_log = []
        
    def log_prediction(self, input_data: Dict[str, Any], 
                      prediction_result: Dict[str, Any],
                      actual_outcome: Optional[int] = None,
                      prediction_time: Optional[float] = None) -> None:
        """
        Log a prediction for performance tracking.
        
        Args:
            input_data: Input data used for prediction
            prediction_result: Prediction result
            actual_outcome: Actual outcome if known
            prediction_time: Time taken for prediction in seconds
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_result.get('prediction'),
            'probability': prediction_result.get('probability', {}).get('advanced_hiv'),
            'risk_level': prediction_result.get('risk_level'),
            'confidence': prediction_result.get('confidence'),
            'actual_outcome': actual_outcome,
            'prediction_time': prediction_time,
            'input_summary': {
                'age': input_data.get('age'),
                'cd4_count': input_data.get('cd4_count'),
                'viral_load': input_data.get('viral_load'),
                'art_adherence': input_data.get('art_adherence')
            }
        }
        
        self.predictions_log.append(log_entry)
        
        # Write to file if specified
        if self.log_file:
            self._write_to_file(log_entry)
    
    def _write_to_file(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance summary for the last N days.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Performance summary dictionary
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter recent predictions
        recent_predictions = [
            entry for entry in self.predictions_log
            if datetime.fromisoformat(entry['timestamp']) > cutoff_date
        ]
        
        if not recent_predictions:
            return {'message': 'No recent predictions found'}
        
        # Calculate summary statistics
        total_predictions = len(recent_predictions)
        
        # Predictions with actual outcomes
        predictions_with_outcomes = [
            entry for entry in recent_predictions
            if entry['actual_outcome'] is not None
        ]
        
        summary = {
            'period_days': days,
            'total_predictions': total_predictions,
            'predictions_with_outcomes': len(predictions_with_outcomes),
            'average_confidence': np.mean([entry['confidence'] for entry in recent_predictions if entry['confidence'] is not None]),
            'risk_level_distribution': self._calculate_risk_distribution(recent_predictions),
            'prediction_distribution': self._calculate_prediction_distribution(recent_predictions)
        }
        
        # Calculate accuracy if we have actual outcomes
        if predictions_with_outcomes:
            correct_predictions = sum(
                1 for entry in predictions_with_outcomes
                if entry['prediction'] == entry['actual_outcome']
            )
            summary['accuracy'] = correct_predictions / len(predictions_with_outcomes)
            
            # Calculate other metrics
            y_true = [entry['actual_outcome'] for entry in predictions_with_outcomes]
            y_pred = [entry['prediction'] for entry in predictions_with_outcomes]
            y_prob = [entry['probability'] for entry in predictions_with_outcomes if entry['probability'] is not None]
            
            if y_prob:
                from sklearn.metrics import roc_auc_score
                try:
                    summary['roc_auc'] = roc_auc_score(y_true, y_prob)
                except ValueError:
                    summary['roc_auc'] = None
        
        return summary
    
    def _calculate_risk_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of risk levels."""
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        for entry in predictions:
            risk_level = entry.get('risk_level')
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        return risk_counts
    
    def _calculate_prediction_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of predictions."""
        pred_counts = {'no_advanced_hiv': 0, 'advanced_hiv': 0}
        for entry in predictions:
            prediction = entry.get('prediction')
            if prediction == 0:
                pred_counts['no_advanced_hiv'] += 1
            elif prediction == 1:
                pred_counts['advanced_hiv'] += 1
        return pred_counts


def load_model_for_inference(model_dir: str) -> HIVModelPredictor:
    """
    Convenience function to load a model for inference.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded model predictor
    """
    return HIVModelPredictor(model_dir)


def validate_model_artifacts(model_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that all required model artifacts are present.
    
    Args:
        model_dir: Directory to check
        
    Returns:
        Tuple of (is_valid, missing_files)
    """
    model_dir = Path(model_dir)
    
    required_files = [
        'hiv_xgboost_model.pkl',
        'feature_columns.pkl'
    ]
    
    optional_files = [
        'scaler.pkl',
        'label_encoder_gender.pkl',
        'model_metadata.json'
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required files
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_required.append(file_name)
    
    # Check optional files
    for file_name in optional_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_optional.append(file_name)
    
    is_valid = len(missing_required) == 0
    
    all_missing = missing_required + missing_optional
    
    if missing_optional:
        logger.warning(f"Optional files missing: {missing_optional}")
    
    return is_valid, all_missing


def create_sample_input() -> Dict[str, Any]:
    """
    Create a sample input for testing the model.
    
    Returns:
        Sample patient data dictionary
    """
    return {
        'age': 45,
        'viral_load': 50000,
        'time_since_diagnosis': 5,
        'art_adherence': 0.6,
        'bmi': 22.5,
        'hemoglobin': 10.5,
        'albumin': 3.0,
        'gender': 'M',
        'comorbidities': 'Diabetes'
    }


def main():
    """Main function for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test HIV model utilities')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model artifacts')
    parser.add_argument('--test_prediction', action='store_true',
                       help='Test single prediction')
    parser.add_argument('--validate_artifacts', action='store_true',
                       help='Validate model artifacts')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.validate_artifacts:
        is_valid, missing_files = validate_model_artifacts(args.model_dir)
        print(f"Model artifacts valid: {is_valid}")
        if missing_files:
            print(f"Missing files: {missing_files}")
    
    if args.test_prediction:
        try:
            # Load model
            predictor = load_model_for_inference(args.model_dir)
            
            # Create sample input
            sample_data = create_sample_input()
            print(f"Sample input: {sample_data}")
            
            # Validate input
            is_valid, errors = ModelValidator.validate_patient_data(sample_data)
            print(f"Input valid: {is_valid}")
            if errors:
                print(f"Validation errors: {errors}")
                return
            
            # Make prediction
            result = predictor.predict(sample_data)
            print(f"Prediction result: {result}")
            
            # Validate result
            is_valid, errors = ModelValidator.validate_prediction_result(result)
            print(f"Result valid: {is_valid}")
            if errors:
                print(f"Result validation errors: {errors}")
            
            # Get model info
            model_info = predictor.get_model_info()
            print(f"Model info: {model_info}")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


if __name__ == "__main__":
    main()
