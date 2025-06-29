"""
HIV Disease Prediction - Model Evaluation Module

This module provides comprehensive model evaluation functionality including
performance metrics, statistical tests, model comparison, and evaluation
reporting for HIV disease prediction models.
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

# ML and statistics imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, calibration_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)


class HIVModelEvaluator:
    """
    Comprehensive model evaluator for HIV disease prediction models.
    
    Features:
    - Multiple evaluation metrics
    - Statistical significance testing
    - Model comparison capabilities
    - Calibration analysis
    - Fairness and bias evaluation
    - Interactive visualizations
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        self.comparison_results = {}
        self.calibration_results = {}
        
    def load_model_artifacts(self, model_dir: str) -> Dict[str, Any]:
        """
        Load model and associated artifacts.
        
        Args:
            model_dir: Directory containing model artifacts
            
        Returns:
            Dictionary with loaded artifacts
        """
        logger.info(f"Loading model artifacts from {model_dir}")
        
        model_dir = Path(model_dir)
        artifacts = {}
        
        # Load model
        model_path = model_dir / 'hiv_xgboost_model.pkl'
        if model_path.exists():
            artifacts['model'] = joblib.load(model_path)
            logger.info("Model loaded successfully")
        
        # Load scaler
        scaler_path = model_dir / 'scaler.pkl'
        if scaler_path.exists():
            artifacts['scaler'] = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        
        # Load encoders
        encoder_path = model_dir / 'label_encoder_gender.pkl'
        if encoder_path.exists():
            artifacts['label_encoder'] = joblib.load(encoder_path)
            logger.info("Label encoder loaded successfully")
        
        # Load feature columns
        features_path = model_dir / 'feature_columns.pkl'
        if features_path.exists():
            artifacts['feature_columns'] = joblib.load(features_path)
            logger.info("Feature columns loaded successfully")
        
        # Load metadata
        metadata_path = model_dir / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                artifacts['metadata'] = json.load(f)
            logger.info("Model metadata loaded successfully")
        
        return artifacts
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Calculating comprehensive evaluation metrics...")
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['specificity'] = self._calculate_specificity(y_true, y_pred)
        
        # Probability-based metrics
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        
        # Additional metrics
        metrics['balanced_accuracy'] = self._calculate_balanced_accuracy(y_true, y_pred)
        metrics['matthews_corrcoef'] = self._calculate_mcc(y_true, y_pred)
        metrics['youden_index'] = metrics['recall'] + metrics['specificity'] - 1
        
        # Clinical metrics
        metrics['positive_predictive_value'] = metrics['precision']
        metrics['negative_predictive_value'] = self._calculate_npv(y_true, y_pred)
        metrics['positive_likelihood_ratio'] = self._calculate_plr(y_true, y_pred)
        metrics['negative_likelihood_ratio'] = self._calculate_nlr(y_true, y_pred)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        logger.info("Comprehensive metrics calculation completed")
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        recall = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (recall + specificity) / 2
    
    def _calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews Correlation Coefficient."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Negative Predictive Value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def _calculate_plr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Positive Likelihood Ratio."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return sensitivity / (1 - specificity) if specificity < 1 else float('inf')
    
    def _calculate_nlr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Negative Likelihood Ratio."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (1 - sensitivity) / specificity if specificity > 0 else float('inf')
    
    def evaluate_calibration(self, y_true: np.ndarray, 
                           y_pred_proba: np.ndarray, 
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        Evaluate model calibration.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration results
        """
        logger.info("Evaluating model calibration...")
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        # Calculate Brier score
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        max_calibration_error = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Hosmer-Lemeshow test
        hl_statistic, hl_p_value = self._hosmer_lemeshow_test(y_true, y_pred_proba, n_bins)
        
        calibration_results = {
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'max_calibration_error': max_calibration_error,
            'hosmer_lemeshow_statistic': hl_statistic,
            'hosmer_lemeshow_p_value': hl_p_value,
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'n_bins': n_bins
        }
        
        self.calibration_results = calibration_results
        logger.info("Model calibration evaluation completed")
        return calibration_results
    
    def _hosmer_lemeshow_test(self, y_true: np.ndarray, 
                            y_pred_proba: np.ndarray, 
                            n_bins: int = 10) -> Tuple[float, float]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Create bins based on predicted probabilities
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        hl_statistic = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find observations in this bin
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
            
            if bin_upper == 1.0:  # Include the upper boundary for the last bin
                in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba <= bin_upper)
            
            if np.sum(in_bin) == 0:
                continue
            
            # Observed and expected counts
            observed_pos = np.sum(y_true[in_bin])
            observed_neg = np.sum(in_bin) - observed_pos
            expected_pos = np.sum(y_pred_proba[in_bin])
            expected_neg = np.sum(in_bin) - expected_pos
            
            # Add to test statistic
            if expected_pos > 0:
                hl_statistic += (observed_pos - expected_pos) ** 2 / expected_pos
            if expected_neg > 0:
                hl_statistic += (observed_neg - expected_neg) ** 2 / expected_neg
        
        # Calculate p-value (chi-square distribution with n_bins-2 degrees of freedom)
        p_value = 1 - stats.chi2.cdf(hl_statistic, n_bins - 2)
        
        return hl_statistic, p_value
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Define scoring metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
            cv_results[f'{metric}_scores'] = scores.tolist()
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
            cv_results[f'{metric}_ci_lower'] = scores.mean() - 1.96 * scores.std() / np.sqrt(cv_folds)
            cv_results[f'{metric}_ci_upper'] = scores.mean() + 1.96 * scores.std() / np.sqrt(cv_folds)
        
        logger.info("Cross-validation completed")
        return cv_results
    
    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray, 
                         sensitive_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate model fairness across different groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            sensitive_features: Dictionary of sensitive feature arrays
            
        Returns:
            Dictionary with fairness metrics
        """
        logger.info("Evaluating model fairness...")
        
        fairness_results = {}
        
        for feature_name, feature_values in sensitive_features.items():
            logger.info(f"Evaluating fairness for {feature_name}")
            
            feature_results = {}
            unique_values = np.unique(feature_values)
            
            # Calculate metrics for each group
            group_metrics = {}
            for value in unique_values:
                mask = feature_values == value
                if np.sum(mask) == 0:
                    continue
                
                group_metrics[str(value)] = {
                    'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                    'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
                    'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
                    'f1_score': f1_score(y_true[mask], y_pred[mask], zero_division=0),
                    'roc_auc': roc_auc_score(y_true[mask], y_pred_proba[mask]) if len(np.unique(y_true[mask])) > 1 else 0,
                    'sample_size': int(np.sum(mask)),
                    'positive_rate': float(np.mean(y_pred[mask])),
                    'true_positive_rate': float(np.mean(y_true[mask]))
                }
            
            feature_results['group_metrics'] = group_metrics
            
            # Calculate fairness metrics
            if len(group_metrics) >= 2:
                # Demographic parity difference
                positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
                feature_results['demographic_parity_difference'] = max(positive_rates) - min(positive_rates)
                
                # Equalized odds difference
                tpr_values = [metrics['recall'] for metrics in group_metrics.values()]
                feature_results['equalized_odds_difference'] = max(tpr_values) - min(tpr_values)
                
                # Equal opportunity difference (same as equalized odds for binary classification)
                feature_results['equal_opportunity_difference'] = feature_results['equalized_odds_difference']
            
            fairness_results[feature_name] = feature_results
        
        logger.info("Fairness evaluation completed")
        return fairness_results
    
    def compare_models(self, models_data: List[Dict[str, Any]], 
                      test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Compare multiple models on the same test data.
        
        Args:
            models_data: List of dictionaries with model info
            test_data: Tuple of (X_test, y_test)
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(models_data)} models...")
        
        X_test, y_test = test_data
        comparison_results = {
            'models': {},
            'rankings': {},
            'statistical_tests': {}
        }
        
        # Evaluate each model
        model_predictions = {}
        for i, model_data in enumerate(models_data):
            model_name = model_data.get('name', f'Model_{i+1}')
            model = model_data['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            comparison_results['models'][model_name] = {
                'metrics': metrics,
                'model_info': model_data.get('info', {})
            }
            
            model_predictions[model_name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        # Rank models by different metrics
        ranking_metrics = ['roc_auc', 'f1_score', 'accuracy', 'average_precision']
        for metric in ranking_metrics:
            model_scores = [(name, data['metrics'][metric]) 
                          for name, data in comparison_results['models'].items()]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            comparison_results['rankings'][metric] = [name for name, score in model_scores]
        
        # Statistical significance tests
        if len(models_data) == 2:
            model_names = list(model_predictions.keys())
            model1_proba = model_predictions[model_names[0]]['y_pred_proba']
            model2_proba = model_predictions[model_names[1]]['y_pred_proba']
            
            # McNemar's test for comparing predictions
            model1_pred = model_predictions[model_names[0]]['y_pred']
            model2_pred = model_predictions[model_names[1]]['y_pred']
            
            mcnemar_stat, mcnemar_p = self._mcnemar_test(y_test, model1_pred, model2_pred)
            
            comparison_results['statistical_tests']['mcnemar'] = {
                'statistic': mcnemar_stat,
                'p_value': mcnemar_p,
                'models_compared': model_names
            }
        
        self.comparison_results = comparison_results
        logger.info("Model comparison completed")
        return comparison_results
    
    def _mcnemar_test(self, y_true: np.ndarray, y_pred1: np.ndarray, 
                     y_pred2: np.ndarray) -> Tuple[float, float]:
        """
        Perform McNemar's test for comparing two models.
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # Count cases
        both_correct = np.sum(correct1 & correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic
        if model1_only + model2_only == 0:
            return 0.0, 1.0
        
        mcnemar_stat = (abs(model1_only - model2_only) - 1) ** 2 / (model1_only + model2_only)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        
        return mcnemar_stat, p_value
    
    def create_evaluation_report(self, model_name: str, 
                               evaluation_results: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            model_name: Name of the model
            evaluation_results: Dictionary with evaluation results
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        logger.info("Creating evaluation report...")
        
        report_lines = []
        report_lines.append(f"# HIV Disease Prediction Model Evaluation Report")
        report_lines.append(f"## Model: {model_name}")
        report_lines.append(f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Performance metrics
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            report_lines.append("## Performance Metrics")
            report_lines.append("")
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")
            
            key_metrics = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall (Sensitivity)', 'recall'),
                ('Specificity', 'specificity'),
                ('F1-Score', 'f1_score'),
                ('ROC-AUC', 'roc_auc'),
                ('Average Precision', 'average_precision'),
                ('Matthews Correlation Coefficient', 'matthews_corrcoef'),
                ('Balanced Accuracy', 'balanced_accuracy')
            ]
            
            for metric_name, metric_key in key_metrics:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    report_lines.append(f"| {metric_name} | {value:.4f} |")
            
            report_lines.append("")
        
        # Confusion matrix
        if 'confusion_matrix' in evaluation_results:
            cm = evaluation_results['confusion_matrix']
            report_lines.append("## Confusion Matrix")
            report_lines.append("")
            report_lines.append("| | Predicted Negative | Predicted Positive |")
            report_lines.append("|---|---|---|")
            report_lines.append(f"| **Actual Negative** | {cm[0][0]} | {cm[0][1]} |")
            report_lines.append(f"| **Actual Positive** | {cm[1][0]} | {cm[1][1]} |")
            report_lines.append("")
        
        # Cross-validation results
        if 'cross_validation' in evaluation_results:
            cv_results = evaluation_results['cross_validation']
            report_lines.append("## Cross-Validation Results")
            report_lines.append("")
            report_lines.append("| Metric | Mean | Std | 95% CI |")
            report_lines.append("|--------|------|-----|--------|")
            
            cv_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            for metric in cv_metrics:
                if f'{metric}_mean' in cv_results:
                    mean_val = cv_results[f'{metric}_mean']
                    std_val = cv_results[f'{metric}_std']
                    ci_lower = cv_results[f'{metric}_ci_lower']
                    ci_upper = cv_results[f'{metric}_ci_upper']
                    report_lines.append(
                        f"| {metric.upper()} | {mean_val:.4f} | {std_val:.4f} | "
                        f"[{ci_lower:.4f}, {ci_upper:.4f}] |"
                    )
            report_lines.append("")
        
        # Calibration results
        if 'calibration' in evaluation_results:
            cal_results = evaluation_results['calibration']
            report_lines.append("## Model Calibration")
            report_lines.append("")
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")
            report_lines.append(f"| Brier Score | {cal_results['brier_score']:.4f} |")
            report_lines.append(f"| Calibration Error | {cal_results['calibration_error']:.4f} |")
            report_lines.append(f"| Max Calibration Error | {cal_results['max_calibration_error']:.4f} |")
            report_lines.append(f"| Hosmer-Lemeshow p-value | {cal_results['hosmer_lemeshow_p_value']:.4f} |")
            report_lines.append("")
        
        # Feature importance
        if 'feature_importance' in evaluation_results:
            feat_imp = evaluation_results['feature_importance']
            report_lines.append("## Top 10 Most Important Features")
            report_lines.append("")
            report_lines.append("| Rank | Feature | Importance |")
            report_lines.append("|------|---------|------------|")
            
            for i, (feature, importance) in enumerate(list(feat_imp.items())[:10], 1):
                report_lines.append(f"| {i} | {feature} | {importance:.4f} |")
            report_lines.append("")
        
        # Clinical interpretation
        report_lines.append("## Clinical Interpretation")
        report_lines.append("")
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            sensitivity = metrics.get('recall', 0)
            specificity = metrics.get('specificity', 0)
            ppv = metrics.get('precision', 0)
            npv = metrics.get('negative_predictive_value', 0)
            
            report_lines.append(f"- **Sensitivity (Recall)**: {sensitivity:.1%} of patients with advanced HIV are correctly identified")
            report_lines.append(f"- **Specificity**: {specificity:.1%} of patients without advanced HIV are correctly identified")
            report_lines.append(f"- **Positive Predictive Value**: {ppv:.1%} of positive predictions are correct")
            report_lines.append(f"- **Negative Predictive Value**: {npv:.1%} of negative predictions are correct")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            auc = metrics.get('roc_auc', 0)
            
            if auc >= 0.9:
                report_lines.append("- **Excellent model performance** - Ready for clinical validation")
            elif auc >= 0.8:
                report_lines.append("- **Good model performance** - Consider additional validation")
            elif auc >= 0.7:
                report_lines.append("- **Fair model performance** - May need improvement")
            else:
                report_lines.append("- **Poor model performance** - Requires significant improvement")
            
            if 'calibration' in evaluation_results:
                hl_p = evaluation_results['calibration']['hosmer_lemeshow_p_value']
                if hl_p < 0.05:
                    report_lines.append("- **Model calibration needs improvement** - Consider calibration techniques")
                else:
                    report_lines.append("- **Model is well-calibrated** - Probabilities are reliable")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        logger.info("Evaluation report created")
        return report_text
    
    def create_interactive_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, 
                               save_path: Optional[str] = None) -> None:
        """
        Create interactive evaluation plots using Plotly.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plots
        """
        logger.info("Creating interactive evaluation plots...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Calibration Plot', 'Prediction Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={auc_score:.3f})',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap_score = average_precision_score(y_true, y_pred_proba)
        
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name=f'PR (AP={ap_score:.3f})',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Calibration Plot
        if hasattr(self, 'calibration_results') and self.calibration_results:
            cal_results = self.calibration_results
            fig.add_trace(
                go.Scatter(x=cal_results['mean_predicted_value'], 
                          y=cal_results['fraction_of_positives'],
                          name='Calibration',
                          line=dict(color='purple', width=2)),
                row=2, col=1
            )
            
            # Perfect calibration line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name='Perfect Calibration',
                          line=dict(color='red', dash='dash')),
                row=2, col=1
            )
        
        # Prediction Distribution
        fig.add_trace(
            go.Histogram(x=y_pred_proba[y_true == 0], name='Negative Class',
                        opacity=0.7, nbinsx=30),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(x=y_pred_proba[y_true == 1], name='Positive Class',
                        opacity=0.7, nbinsx=30),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="HIV Disease Prediction Model Evaluation",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Mean Predicted Value", row=2, col=1)
        fig.update_yaxes(title_text="Fraction of Positives", row=2, col=1)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Save plot if path provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plot_file = Path(save_path) / 'interactive_evaluation.html'
            fig.write_html(plot_file)
            logger.info(f"Interactive plots saved to {plot_file}")
        
        fig.show()
        logger.info("Interactive evaluation plots created")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate HIV disease prediction model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model artifacts')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data CSV file')
    parser.add_argument('--output', type=str, default='evaluation_results/',
                       help='Output directory for evaluation results')
    parser.add_argument('--report', action='store_true',
                       help='Generate evaluation report')
    parser.add_argument('--plots', action='store_true',
                       help='Generate evaluation plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = HIVModelEvaluator()
    
    # Load model artifacts
    artifacts = evaluator.load_model_artifacts(args.model_dir)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    # Prepare test data (assuming preprocessing is already done)
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
    
    # Generate report if requested
    if args.report:
        report = evaluator.create_evaluation_report(
            'HIV_XGBoost_Model', 
            evaluation_results,
            save_path=f"{args.output}/evaluation_report.md"
        )
        print("Evaluation report generated")
    
    # Generate plots if requested
    if args.plots:
        evaluator.create_interactive_plots(
            y_test, y_pred, y_pred_proba,
            save_path=args.output
        )
    
    # Save evaluation results
    Path(args.output).mkdir(parents=True, exist_ok=True)
    results_file = Path(args.output) / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\nModel Evaluation Complete!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
