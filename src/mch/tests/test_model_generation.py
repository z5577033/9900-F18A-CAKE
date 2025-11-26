import unittest
from unittest.mock import MagicMock
import polars as pl
import sys
import os

# 1. Ensure src can be imported
project_path = "/Workspace/9900-f18a-cake/mt-method2/src"
if project_path not in sys.path:
    sys.path.append(project_path)

from mch.models.training import BatchModelTrainer

# --- Simple Mock Tree Node (simulate DiseaseTree) ---
class MockNode:
    def __init__(self, name, children=None, samples=None):
        self.name = name
        self.children = children or []
        self.samples = samples or [] 

    def get_child_names(self):
        return [c.name for c in self.children]

    def find_node_by_name(self, name):
        if self.name == name: return self
        for c in self.children:
            res = c.find_node_by_name(name)
            if res: return res
        return None

    def get_samples_recursive(self):
        all_samples = set(self.samples)
        for c in self.children:
            all_samples.update(c.get_samples_recursive())
        return list(all_samples)

class TestModelGeneration(unittest.TestCase):
    """
    Test core logic of model training (BatchModelTrainer)
    Focus: normal training, rare class removal, skip single class
    """

    def setUp(self):
        # 1. Build a fake tree: Root -> [TypeA (10 samples), TypeB (10 samples)]
        self.node_a = MockNode("TypeA", samples=[f"s{i}" for i in range(10)])
        self.node_b = MockNode("TypeB", samples=[f"s{i}" for i in range(10, 20)])
        self.root_node = MockNode("RootDisease", children=[self.node_a, self.node_b])
        
        # 2. Create a tiny fake dataset (20 rows, 2 features)
        # Feature 1 is highly discriminative, feature 2 is noise
        ids = [f"s{i}" for i in range(20)]
        feat1 = [0.1] * 10 + [0.9] * 10 
        feat2 = [0.5] * 20
        
        self.mock_data = pl.DataFrame({
            "biosample_id": ids,
            "feat1": feat1,
            "feat2": feat2
        })

        # 3. Initialize Trainer
        self.trainer = BatchModelTrainer(tree=self.root_node)
        self.trainer.filteredMValueFile = self.mock_data
        
        # 4. Config params (force only train RootDisease, only run 2 trees)
        os.environ["MCH_ONLY_NODE"] = "RootDisease"
        self.trainer.rf_params = {"n_estimators": 2, "max_depth": 2}
        self.trainer.prefilter_topk = 2

    def test_successful_training(self):
        """Test 1: Standard flow (sufficient data, should train successfully)"""
        print("\n🧪 Test: Simulating standard training...")
        results = self.trainer.train_all_models(raise_on_error=True)
        
        self.assertIn("RootDisease", results, "Result should contain target node")
        metrics = results["RootDisease"]["metrics"]
        print(f"   🏆 Simulated Accuracy: {metrics['accuracy']}")
        
        # This simple data should easily get 1.0
        self.assertGreater(metrics["accuracy"], 0.0)

    def test_auto_drop_rare_classes(self):
        """Test 2: Auto drop rare class (Critical Fix validation)"""
        print("\n🧪 Test: Simulating rare class drop...")
        
        # Add a troublemaker TypeC with only 1 sample
        node_c = MockNode("TypeC", samples=["s999"])
        self.root_node.children.append(node_c)
        
        # Add this row to the data as well
        row_c = pl.DataFrame({"biosample_id": ["s999"], "feat1": [0.5], "feat2": [0.5]})
        self.trainer.filteredMValueFile = self.mock_data.vstack(row_c)
        
        # Expect: code auto drops TypeC and trains successfully, not ValueError
        try:
            results = self.trainer.train_all_models(raise_on_error=True)
            print("   ✅ Passed! Rare class was handled gracefully.")
            self.assertIn("RootDisease", results)
        except ValueError as e:
            self.fail(f"❌ Trainer crashed on rare class! Error: {e}")

    def test_skip_if_single_class(self):
        """Test 3: Skip if only one class exists"""
        print("\n🧪 Test: Simulating single-class skip...")
        
        # Remove TypeB, only TypeA left (cannot do binary classification)
        self.root_node.children = [self.node_a]
        
        results = self.trainer.train_all_models()
        
        # Expect: result is empty or does not contain RootDisease
        self.assertNotIn("RootDisease", results, "Should skip if only 1 subclass exists")
        print("   ✅ Passed! Correctly skipped single-class node.")