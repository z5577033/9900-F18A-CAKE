#!/usr/bin/env python3
"""
Example script demonstrating standardized model training and saving.

This script shows how to train models using the enhanced research-grade
model persistence system that saves:
- model.joblib (the trained model)
- params.json (training parameters and configuration)
- metrics.json (performance metrics and evaluation results)
- metadata.json (comprehensive metadata for reproducibility)

Run this script to train models with full audit trail and reproducibility.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mch.models.model_generation import generate_model_for_node
from mch.utils.model_persistence import load_research_model
from mch.config.settings import main_tree, FREEZE_DIR
from mch.utils.logging_utils import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Main training pipeline with standardized model saving.
    """
    logger.info("Starting research-grade model training pipeline")
    logger.info(f"Using data from: {FREEZE_DIR}")
    
    # Create results directory
    results_dir = Path("results/models_research_standard")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if main_tree is None:
        logger.error("Main tree not loaded. Check data files and configuration.")
        return
    
    logger.info(f"Loaded disease tree: {main_tree.name}")
    logger.info(f"Tree has {len(main_tree.children) if hasattr(main_tree, 'children') else 'unknown'} child nodes")
    
    # Train models for each node in the tree
    trained_models = []
    
    try:
        # Train model for root node
        logger.info(f"Training model for root node: {main_tree.name}")
        saved_model_dir = generate_model_for_node(main_tree, results_dir)
        
        if saved_model_dir:
            trained_models.append(saved_model_dir)
            logger.info(f"Successfully trained and saved model for {main_tree.name}")
            
            # Demonstrate loading the saved model
            logger.info("Demonstrating model loading...")
            loaded_model_package = load_research_model(saved_model_dir)
            
            logger.info("Loaded model package contains:")
            for key in loaded_model_package.keys():
                logger.info(f"  - {key}")
            
            # Print some key metrics
            if 'metrics' in loaded_model_package:
                metrics = loaded_model_package['metrics']
                logger.info(f"Model accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                logger.info(f"Number of test samples: {metrics.get('n_test_samples', 'N/A')}")
                logger.info(f"Number of features: {metrics.get('n_features', 'N/A')}")
        
        # Train models for child nodes if they exist
        if hasattr(main_tree, 'children') and main_tree.children:
            for child_node in main_tree.children:
                try:
                    logger.info(f"Training model for child node: {child_node.name}")
                    saved_model_dir = generate_model_for_node(child_node, results_dir)
                    
                    if saved_model_dir:
                        trained_models.append(saved_model_dir)
                        logger.info(f"Successfully trained and saved model for {child_node.name}")
                
                except Exception as e:
                    logger.error(f"Failed to train model for {child_node.name}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    # Summary
    logger.info(f"Training completed. Successfully trained {len(trained_models)} models:")
    for model_dir in trained_models:
        logger.info(f"  - {model_dir}")
    
    logger.info("All models saved with standardized format:")
    logger.info("  - model.joblib (trained model)")
    logger.info("  - params.json (training parameters)")
    logger.info("  - metrics.json (performance metrics)")
    logger.info("  - metadata.json (comprehensive metadata)")
    logger.info("  - features.json (feature information)")
    logger.info("  - test_predictions.csv (test set predictions)")

def demonstrate_model_audit():
    """
    Demonstrate how to audit and inspect saved models.
    """
    logger.info("=" * 60)
    logger.info("DEMONSTRATING MODEL AUDIT CAPABILITIES")
    logger.info("=" * 60)
    
    results_dir = Path("results/models_research_standard")
    
    # Find all saved models
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    for model_dir in model_dirs:
        logger.info(f"\nAuditing model: {model_dir.name}")
        logger.info("-" * 40)
        
        try:
            # Load complete model package
            package = load_research_model(model_dir)
            
            # Display metadata
            if 'metadata' in package:
                metadata = package['metadata']
                logger.info(f"Model type: {metadata.get('model_type', 'Unknown')}")
                logger.info(f"Created: {metadata.get('creation_timestamp', 'Unknown')}")
                logger.info(f"Classes: {metadata.get('data_info', {}).get('n_classes', 'Unknown')}")
                logger.info(f"Features: {metadata.get('data_info', {}).get('n_features', 'Unknown')}")
            
            # Display key metrics
            if 'metrics' in package:
                metrics = package['metrics']
                logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
                logger.info(f"Macro F1: {metrics.get('macro_f1', 'N/A'):.3f}")
                logger.info(f"Weighted F1: {metrics.get('weighted_f1', 'N/A'):.3f}")
            
            # Display training parameters
            if 'params' in package:
                params = package['params']
                logger.info(f"Training time: {params.get('training_timestamp', 'Unknown')}")
                if 'training_config' in params:
                    training_config = params['training_config']
                    logger.info(f"Best CV score: {training_config.get('best_cv_score', 'N/A'):.3f}")
                    logger.info(f"Training duration: {training_config.get('training_duration_seconds', 'N/A'):.1f}s")
        
        except Exception as e:
            logger.error(f"Error auditing model {model_dir.name}: {e}")

if __name__ == "__main__":
    try:
        main()
        demonstrate_model_audit()
        logger.info("Research-grade model training completed successfully!")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)