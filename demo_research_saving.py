#!/usr/bin/env python3
"""
Demonstration of research-grade model saving system.

This script creates a mock model training scenario to demonstrate
the standardized saving format that includes:
- model.joblib
- params.json  
- metrics.json
- metadata.json
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mch.utils.model_persistence import save_research_model, load_research_model

def create_demo_data():
    """Create synthetic methylation-like data for demonstration."""
    print("Creating synthetic methylation data for demonstration...")
    
    # Generate synthetic data similar to methylation profiles
    X, y = make_classification(
        n_samples=500,
        n_features=1000,
        n_informative=100,
        n_redundant=50,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    
    # Create feature names like CpG sites
    feature_names = [f"cg{str(i).zfill(8)}" for i in range(X.shape[1])]
    
    # Create class names like cancer types  
    class_names = ["Glioblastoma", "Breast_Cancer", "Lung_Cancer", "Leukemia"]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series([class_names[i] for i in y], name="cancer_type")
    
    print(f"✓ Created dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features, {len(class_names)} classes")
    return X_df, y_series, feature_names, class_names

def train_demo_model(X, y, feature_names, class_names):
    """Train a demonstration model."""
    print("Training demonstration model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    print(f"✓ Model trained on {len(X_train)} samples")
    print(f"✓ Test set has {len(X_test)} samples")
    
    return pipeline, X_test, y_test

def demonstrate_research_saving():
    """Demonstrate the research-grade model saving system."""
    print("="*60)
    print("DEMONSTRATING RESEARCH-GRADE MODEL SAVING")
    print("="*60)
    
    # Create demonstration data
    X, y, feature_names, class_names = create_demo_data()
    
    # Train model
    model, X_test, y_test = train_demo_model(X, y, feature_names, class_names)
    
    # Prepare training parameters
    training_params = {
        "model_type": "random_forest",
        "dataset_type": "synthetic_methylation",
        "preprocessing": {
            "scaling": "StandardScaler",
            "feature_selection": "none"
        },
        "cross_validation": {
            "method": "none",
            "note": "Demo model - no CV performed"
        }
    }
    
    # Additional metadata
    additional_metadata = {
        "experiment_type": "demonstration",
        "data_source": "synthetic",
        "research_question": "Demonstrate standardized model saving",
        "notes": "This is a demonstration of the research-grade model persistence system"
    }
    
    # Save the model using research standards
    print("\nSaving model with research standards...")
    results_dir = Path("results/demo_research_models")
    
    saved_model_dir = save_research_model(
        model=model,
        model_name="demo_methylation_classifier",
        X_test=X_test,
        y_test=y_test,
        save_directory=results_dir,
        training_params=training_params,
        feature_names=feature_names,
        class_names=class_names,
        additional_metadata=additional_metadata
    )
    
    print(f"✓ Model saved to: {saved_model_dir}")
    
    # Demonstrate loading and inspection
    print("\nDemonstrating model loading and inspection...")
    loaded_package = load_research_model(saved_model_dir)
    
    print("Loaded package contains:")
    for key in loaded_package.keys():
        print(f"  ✓ {key}")
    
    # Display key information
    if 'metadata' in loaded_package:
        metadata = loaded_package['metadata']
        print(f"\nModel Metadata:")
        print(f"  Name: {metadata.get('model_name')}")
        print(f"  Type: {metadata.get('model_type')}")
        print(f"  Created: {metadata.get('creation_timestamp')}")
        print(f"  Classes: {metadata.get('data_info', {}).get('n_classes')}")
        print(f"  Features: {metadata.get('data_info', {}).get('n_features')}")
    
    if 'metrics' in loaded_package:
        metrics = loaded_package['metrics']
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"  Macro F1: {metrics.get('macro_f1', 0):.3f}")
        print(f"  Weighted F1: {metrics.get('weighted_f1', 0):.3f}")
    
    if 'params' in loaded_package:
        params = loaded_package['params']
        print(f"\nTraining Parameters:")
        print(f"  Model Type: {params.get('model_type')}")
        print(f"  Training Time: {params.get('training_timestamp')}")
    
    # Show file structure
    print(f"\nSaved Files Structure:")
    for file_path in sorted(saved_model_dir.glob("*")):
        print(f"  📄 {file_path.name}")
    
    print("\n" + "="*60)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    print("Your research models will now be saved with:")
    print("✓ model.joblib - The trained model")
    print("✓ params.json - Training parameters and configuration")
    print("✓ metrics.json - Comprehensive performance metrics")  
    print("✓ metadata.json - Complete metadata for reproducibility")
    print("✓ features.json - Feature names and information")
    print("✓ test_predictions.csv - Test set predictions for analysis")
    print("\nThis ensures full audit trail and reproducibility for your research!")

if __name__ == "__main__":
    try:
        demonstrate_research_saving()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)