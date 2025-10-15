"""
Standard model persistence utilities for research reproducibility.

This module provides standardized saving and loading of machine learning models
with comprehensive metadata for audit trails and result reproducibility.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class ModelPersistence:
    """
    Standardized model saving and loading with metadata.
    
    This class ensures that all trained models are saved with:
    - The model itself (model.joblib)
    - Training parameters (params.json)
    - Performance metrics (metrics.json)
    - Metadata for reproducibility
    """
    
    def __init__(self, base_directory: Union[str, Path]):
        """
        Initialize the model persistence handler.
        
        Args:
            base_directory: Base directory for saving models
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
    def save_model_complete(
        self,
        model: BaseEstimator,
        model_name: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        training_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a complete model package with all metadata.
        
        Args:
            model: Trained scikit-learn model
            model_name: Name identifier for the model
            X_test: Test features
            y_test: True test labels
            y_pred: Predicted labels (will be generated if None)
            training_params: Parameters used during training
            feature_names: Names of features used
            class_names: Names of target classes
            additional_metadata: Any additional metadata to store
            
        Returns:
            Path to the created model directory
        """
        # Create model-specific directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = self.base_directory / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model package to: {model_dir}")
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = model.predict(X_test)
            
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to: {model_path}")
        
        # Save parameters
        params_data = self._extract_model_params(model, training_params)
        params_path = model_dir / "params.json"
        with open(params_path, 'w') as f:
            json.dump(params_data, f, indent=2, default=self._json_serializer)
        logger.info(f"Saved parameters to: {params_path}")
        
        # Save metrics
        metrics_data = self._calculate_metrics(y_test, y_pred, X_test, class_names)
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=self._json_serializer)
        logger.info(f"Saved metrics to: {metrics_path}")
        
        # Save metadata
        metadata = self._create_metadata(
            model_name, model, X_test, y_test, feature_names, 
            class_names, additional_metadata
        )
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=self._json_serializer)
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Save feature names if provided
        if feature_names is not None:
            features_path = model_dir / "features.json"
            with open(features_path, 'w') as f:
                json.dump({"feature_names": feature_names}, f, indent=2)
        
        # Save test predictions for later analysis
        predictions_df = pd.DataFrame({
            'true_labels': y_test,
            'predicted_labels': y_pred
        })
        predictions_path = model_dir / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        
        logger.info(f"Model package saved successfully to: {model_dir}")
        return model_dir
    
    def _extract_model_params(self, model: BaseEstimator, training_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Extract model parameters and training configuration."""
        params = {
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "training_timestamp": datetime.now().isoformat(),
        }
        
        if training_params:
            params["training_config"] = training_params
            
        # Extract pipeline parameters if it's a pipeline
        if hasattr(model, 'steps'):
            params["pipeline_steps"] = [step[0] for step in model.steps]
            params["pipeline_params"] = {}
            for step_name, step_model in model.steps:
                params["pipeline_params"][step_name] = step_model.get_params()
        
        return params
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          X_test: pd.DataFrame, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "n_test_samples": len(y_true),
            "n_features": X_test.shape[1],
        }
        
        # Calculate precision, recall, f1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Overall metrics
        metrics["macro_precision"] = float(np.mean(precision))
        metrics["macro_recall"] = float(np.mean(recall))
        metrics["macro_f1"] = float(np.mean(f1))
        
        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics["weighted_precision"] = float(precision_w)
        metrics["weighted_recall"] = float(recall_w)
        metrics["weighted_f1"] = float(f1_w)
        
        # Per-class metrics
        unique_classes = sorted(list(set(y_true)))
        metrics["per_class_metrics"] = {}
        
        for i, class_label in enumerate(unique_classes):
            class_name = class_names[i] if class_names and i < len(class_names) else str(class_label)
            metrics["per_class_metrics"][class_name] = {
                "precision": float(precision[i]) if i < len(precision) else 0.0,
                "recall": float(recall[i]) if i < len(recall) else 0.0,
                "f1_score": float(f1[i]) if i < len(f1) else 0.0,
                "support": int(support[i]) if i < len(support) else 0
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report as dict
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = class_report
        
        return metrics
    
    def _create_metadata(self, model_name: str, model: BaseEstimator, 
                        X_test: pd.DataFrame, y_test: pd.Series,
                        feature_names: Optional[List[str]] = None,
                        class_names: Optional[List[str]] = None,
                        additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive metadata for the model."""
        metadata = {
            "model_name": model_name,
            "creation_timestamp": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "sklearn_version": None,  # Will be filled if sklearn is available
            "data_info": {
                "n_samples": len(y_test),
                "n_features": X_test.shape[1],
                "n_classes": len(set(y_test)),
                "class_distribution": dict(y_test.value_counts()),
            }
        }
        
        # Add sklearn version if available
        try:
            import sklearn
            metadata["sklearn_version"] = sklearn.__version__
        except ImportError:
            pass
        
        # Add feature information
        if feature_names:
            metadata["feature_info"] = {
                "n_features": len(feature_names),
                "feature_names": feature_names[:100]  # Limit to first 100 for readability
            }
        
        # Add class information
        if class_names:
            metadata["class_info"] = {
                "class_names": class_names,
                "n_classes": len(class_names)
            }
        
        # Add any additional metadata
        if additional_metadata:
            metadata["additional"] = additional_metadata
        
        return metadata
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types and sklearn objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # For sklearn objects and other objects with __dict__
            return {
                "_type": type(obj).__name__,
                "_module": type(obj).__module__,
                "_repr": repr(obj)[:200] + "..." if len(repr(obj)) > 200 else repr(obj)
            }
        elif hasattr(obj, '__class__'):
            # For other objects, just store type information
            return {
                "_type": type(obj).__name__,
                "_module": type(obj).__module__,
                "_str": str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
            }
        else:
            # Fallback to string representation
            return str(obj)
    
    def load_model_complete(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a complete model package.
        
        Args:
            model_path: Path to the model directory or model.joblib file
            
        Returns:
            Dictionary containing model, params, metrics, and metadata
        """
        model_path = Path(model_path)
        
        # If it's a file, get the parent directory
        if model_path.is_file():
            model_dir = model_path.parent
            model_file = model_path
        else:
            model_dir = model_path
            model_file = model_dir / "model.joblib"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load all components
        result = {}
        
        # Load model
        result["model"] = joblib.load(model_file)
        
        # Load parameters
        params_file = model_dir / "params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                result["params"] = json.load(f)
        
        # Load metrics
        metrics_file = model_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                result["metrics"] = json.load(f)
        
        # Load metadata
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                result["metadata"] = json.load(f)
        
        # Load features
        features_file = model_dir / "features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                result["features"] = json.load(f)
        
        # Load test predictions
        predictions_file = model_dir / "test_predictions.csv"
        if predictions_file.exists():
            result["test_predictions"] = pd.read_csv(predictions_file)
        
        logger.info(f"Loaded complete model package from: {model_dir}")
        return result


def save_research_model(
    model: BaseEstimator,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_directory: Union[str, Path],
    **kwargs
) -> Path:
    """
    Convenience function to save a model with research standards.
    
    This is a simplified interface to the ModelPersistence class for quick usage.
    
    Args:
        model: Trained model
        model_name: Model identifier
        X_test: Test features
        y_test: Test labels
        save_directory: Directory to save the model
        **kwargs: Additional arguments passed to save_model_complete
        
    Returns:
        Path to the saved model directory
    """
    persistence = ModelPersistence(save_directory)
    return persistence.save_model_complete(
        model=model,
        model_name=model_name,
        X_test=X_test,
        y_test=y_test,
        **kwargs
    )


def load_research_model(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load a research model package.
    
    Args:
        model_path: Path to model directory or model.joblib file
        
    Returns:
        Complete model package with all metadata
    """
    # Create a temporary persistence object just for loading
    temp_persistence = ModelPersistence(Path(model_path).parent)
    return temp_persistence.load_model_complete(model_path)