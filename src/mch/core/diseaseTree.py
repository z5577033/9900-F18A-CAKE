
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DiseaseTree:
    name: str
    children: list
    samples: list
    validationSamples: list = field(default_factory=list)
    #validationSamples: None = None
    classifier: None = None
    selectedFeatures: list = field(default_factory=list)
    #selectedFeatures: None = None


        
    def findSample(self, sample_id: str, path: Optional[List[str]] = None) -> Optional[List[str]]:
        if path is None:
            path = []
        path.append(self.name)
        if sample_id in self.samples:
            return path
        for child in self.children:
            result = findSample(sample_id)
            if result is not None:
                return result
        path.pop()
        return None

    def delete_node(self, target_name):
        if self.name == target_name:
            # If the current node matches the target, return None to delete it
            return None
        
        # Recursively search for the target node in children
        new_children = [child.delete_node(target_name) for child in self.children]
        # Remove None (deleted nodes) from children list
        new_children = [child for child in new_children if child is not None]

        # Update children list with modified list
        self.children = new_children
        
        return self  # Return modified node



    def find_node_by_name(self, target_name):
        #print(target_name)
        if self.name == target_name:
            return self

        for child in self.children:
            result = child.find_node_by_name(target_name)
            if result:
                return result
        return None

    def get_child_names(self):
        return [child.name for child in self.children]

    def get_samples_recursive(self):
        # Start with the samples of the current node
        # Recursively collect samples from children
        collected_samples = self.samples.copy()

        if not self.children:
            return self.samples
        else:
            for child in self.children:
                collected_samples.extend(child.get_samples_recursive())

        return collected_samples


#        return all_samples

    def get_nodes_at_level(self, level: int) -> list['DiseaseTree']:
        # Create a list to store nodes at the specified level
        nodes_at_level = []
        # Recursive function to traverse the tree and collect nodes
        def collect_nodes(node: 'DiseaseTree', current_level: int):
            if current_level == level:
                nodes_at_level.append(node)
            elif current_level < level:
                for child in node.children:
                    collect_nodes(child, current_level + 1)

        # Start the traversal from the root node
        collect_nodes(self, 1)

        return nodes_at_level

    def filter_tree_by_depth(node, target_depth):
        if target_depth == 0:
            return [node.name]
        elif target_depth > 0:
            result = []
            for child in node.children:
                child_result = filter_tree_by_depth(child, target_depth - 1)
                if child_result:
                    result.extend(child_result)
            return result
        else:
            return []

    def build_disease_tree(self, node_name, diseaseTreedf, sampledf):
    #def build_disease_tree(self, node_name):
        name = node_name
        children = list(diseaseTreedf[diseaseTreedf.parent == node_name].child)
        samples = list(sampledf[sampledf.diseaseName == node_name].sample_id)
        
        # Recursively build child nodes
        child_nodes = [self.build_disease_tree(child, diseaseTreedf, sampledf) for child in children]
        #child_nodes = [self.build_disease_tree(child) for child in children]

        # Remove nodes with no samples and no children
        child_nodes = [node for node in child_nodes if node.samples or node.children]
        
        return DiseaseTree(node_name, child_nodes, samples)

    def get_samples_at_level(self, level: int) -> pd.DataFrame:
            # Initialize an empty DataFrame to store the results
            dfs = []

            nodes = self.get_nodes_at_level(level)
            #print(len(nodes))
            for tree in nodes:
                samples = tree.get_samples_recursive()
                df = pd.DataFrame({"sampleId": samples,"cancerType": tree.name})
                dfs.append(df)
            
            results = pd.concat(dfs, ignore_index=True)
            
            return results
