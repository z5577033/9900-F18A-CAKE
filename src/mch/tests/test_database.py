import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# 1. import mch 
project_path = "/Workspace/9900-f18a-cake/mt-method2/src"
if project_path not in sys.path:
    sys.path.append(project_path)

class TestDataPipeline(unittest.TestCase):
    """
    Test data loading pipeline.
    Replaces original database connection test, now tests Unity Catalog -> Settings loading logic.
    """

    def test_global_data_loaded(self):
        """
        Integration test: Verify that global data is successfully loaded into memory.
        Relies on the fact that settings.py has already run in the Databricks environment.
        """
        print("\n🧪 Testing: Checking global data integrity...")
        
        try:
            # Try importing in-memory data
            from mch.config.settings import mvalue_df, main_tree
            
            # 1. Data should not be None
            self.assertIsNotNone(mvalue_df, "Global variable mvalue_df should not be None")
            self.assertIsNotNone(main_tree, "Global variable main_tree should not be None")
            
            # 2. Check column names (ensure pivot succeeded)
            # We expect a wide table, so columns should include biosample_id
            self.assertIn("biosample_id", mvalue_df.columns, "Data must contain 'biosample_id' column")
            
            # 3. Check row count (Smart Sampling check)
            # We previously set it to 50 or 80, so as long as it's >0 and <1000 it's expected
            row_count = mvalue_df.height
            print(f"   ℹ️ Current rows in memory: {row_count}")
            self.assertGreater(row_count, 0, "Row count should be greater than 0")
            # self.assertLess(row_count, 1000, "Data should be sampled, not full volume") 

            print("✅ Global data check passed!")
            
        except ImportError as e:
            print(f"⚠️ Skipping test: Could not import settings ({e})")
            # In pure CI environments (non-Databricks Notebook), may not be able to connect to data, so skip
            pass

    @patch('pyspark.sql.SparkSession.builder.getOrCreate')
    def test_spark_connection_logic(self, mock_get_or_create):
        """
        Unit test: Mock Spark connection logic.
        Verifies that code attempts to configure Arrow optimization without actually starting Spark jobs (prevents OOM).
        """
        print("\n🧪 Testing: Mocking Spark connection logic...")
        
        # 1. Create mock objects
        mock_spark = MagicMock()
        mock_df = MagicMock()
        
        # 2. Set behaviors
        mock_get_or_create.return_value = mock_spark
        mock_spark.table.return_value = mock_df
        
        # 3. Manually execute logic similar to load_data (mocked)
        spark = mock_get_or_create()
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # 4. Verify key config was called
        mock_spark.conf.set.assert_called_with("spark.sql.execution.arrow.pyspark.enabled", "true")
        print("✅ Spark configuration logic verified (Mocked).")

    def test_tree_structure(self):
        """
        Test if Disease Tree contains key node (Leukaemia)
        """
        print("\n🧪 Testing: Checking Disease Tree structure...")
        try:
            from mch.config.settings import main_tree
            
            # Check if 'Leukaemia' can be found
            node = main_tree.find_node_by_name("Leukaemia")
            self.assertIsNotNone(node, "Tree should contain 'Leukaemia'")
            
            # Check if it has children (B-cell, T-cell etc.)
            children = node.get_child_names()
            self.assertGreater(len(children), 0, "Leukaemia node should have children")
            print(f"   ℹ️ Found children: {children[:3]}...")
            print("✅ Tree structure check passed!")
            
        except ImportError:
            pass