import random
import numpy as np
import csv
import json
import hashlib
import re
import math
import operator
import time
import scipy.linalg
import tqdm
# generic node class that can be used to represent a node in a tree
class Node:
    def __init__(self, module_id, module_name, features, feature_names=None, output=None, total_output=None, children=None):
        # id of the module
        self.module_id = module_id

        self.module_name = module_name
        # name of the module
        self.features = features
        # children of the module
        self.children = children if children is not None else []
        # immediate output of the module
        self.output = output
        # total output of the module including the output of its children
        self.total_output = None
        self.left_child = self.children[0] if len(self.children) > 0 else None
        self.right_child = self.children[1] if len(self.children) > 1 else None
        self.left_child_name = self.left_child.module_name if self.left_child else None
        self.right_child_name = self.right_child.module_name if self.right_child else None
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(len(features))]
        else:
            self.feature_names = feature_names


    def to_dict(self):
        node_dict = {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "features": self.features,
            "feature_names": self.feature_names,
            "output": self.output,
            "total_output": self.total_output,
            "left_child_name": self.left_child_name,
            "right_child_name": self.right_child_name,
        }
        if self.children:
            node_dict["children"] = [child.to_dict() for child in self.children]
        else:
            node_dict["children"] = None
        return node_dict
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            module_id=data['module_id'],
            module_name=data['module_name'],
            features=data['features'],
            feature_names=data['feature_names'],
            output=data['output'],
            total_output=data['total_output'],
            children=[cls.from_dict(child) for child in data['children']] if data['children'] else [],
        
        )
    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)
    
    def print_tree(self, level=0):
        indent = "  " * level
        print(f"{indent}Module ID: {self.module_id}")
        print(f"{indent}Features: {self.features}")
        if self.output is not None:
            print(f"{indent}Output: {self.output}")
        if self.children:
            print(f"{indent}Children:")
            for child in self.children:
                child.print_tree(level + 1)
        else:
            print(f"{indent}No children")

    def return_outputs_as_list_postorder(self, include_total_output=False):
        outputs = []
        
        # First, recursively gather outputs from all children
        for child in self.children:
            outputs.extend(child.return_outputs_as_list(include_total_output))
        
        # Then, append the current node's output
        if include_total_output:
            if self.total_output is not None:
                outputs.append(self.total_output)
        else:
            if self.output is not None:
                outputs.append(self.output)
        
        return outputs
    
    def get_depth(self):
        if not self.children:
            return 1
        return 1 + max([child.get_depth() for child in self.children])

    def expand_features(self):
        expanded_features = self.features.copy()

        if self.children == []:
            # expand by adding 0
            expanded_features.append(0.0)
            self.features = expanded_features
            return    

        for child in self.children:
            child.expand_features()
            expanded_features.append(child.output)

        self.features = expanded_features

# let's make ExpressionNode a subclass of Node
class ExpressionNode(Node):
    def __init__(
        self, 
        module_id,
        module_name,
        features,
        feature_names,
        output,
        total_output,
        children
       ):
        super().__init__(module_id, module_name, features, feature_names, output, total_output, children)

        
        self.left_child = self.children[0] if len(children) > 0 else None
        self.right_child = self.children[1] if len(children) > 1 else None
        self.left_child_name = self.left_child.module_name if self.left_child else None
        self.right_child_name = self.right_child.module_name if self.right_child else None
        
        

    def to_dict(self):
        node_dict = {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "features": self.features,
            "feature_names": self.feature_names,
            "output": self.output,
            "total_output": self.total_output,
            "left_child_name": self.left_child_name,
            "right_child_name": self.right_child_name,
        }
        if self.children:
            node_dict["children"] = [child.to_dict() for child in self.children]
        else:
            node_dict["children"] = None
        return node_dict

    def return_outputs_as_list_postorder(self, outcomes_parallel=False):
        outputs = []
        
        if self.left_child:
            outputs.extend(self.left_child.return_outputs_as_list_postorder(outcomes_parallel=outcomes_parallel))
        
        if self.right_child:
            outputs.extend(self.right_child.return_outputs_as_list_postorder(outcomes_parallel=outcomes_parallel))
        
        if outcomes_parallel:
            if self.output_run_time is not None:
                outputs.append(self.output_run_time)
        else:
            if self.output_total_run_time is not None:
                outputs.append(self.output_total_run_time)
        
        return outputs

    def return_outputs_as_list_preorder(self, outcomes_parallel=False):
        outputs = []
        
        if outcomes_parallel:
            if self.output is not None:
                outputs.append(self.output)
        else:
            if self.total_output is not None:
                outputs.append(self.total_output)
        
        if self.left_child:
            outputs.extend(self.left_child.return_outputs_as_list_preorder(outcomes_parallel=outcomes_parallel))
        
        if self.right_child:
            outputs.extend(self.right_child.return_outputs_as_list_preorder(outcomes_parallel=outcomes_parallel))
        
        return outputs

    def get_depth(self):
        if self.left_child is None and self.right_child is None:
            return 1
        left_depth = self.left_child.get_depth() if self.left_child else 0
        right_depth = self.right_child.get_depth() if self.right_child else 0
        return max(left_depth, right_depth) + 1

    
    @classmethod
    def from_dict_custom(cls, data):
        # TODO: Norm features are missing currently, need to add them.
        # Run first set of experiments and then add them.
        module_name_map = {"*": "mult", "-": "sub", "+": "add"}
        module_name = module_name_map.get(data['operator_name'], data['operator_name'])
        module_id = module_name
        left_child_input_shape = data['left_child_input_shape']
        right_child_input_shape = data['right_child_input_shape']
        left_child_norm = data['left_child_norm']
        right_child_norm = data['right_child_norm']
        output_shape = data['output_shape']
        output_norm = data['output_norm']
        features = [left_child_input_shape, left_child_norm, right_child_input_shape, right_child_norm, output_shape, output_norm]
        feature_names = ["left_child_input_shape", "left_child_norm", "right_child_input_shape", "right_child_norm", "output_shape", "output_norm"]
        output = data['output_run_time']
        total_output = data['output_total_run_time']
        left_child_name = data['left_child_name']
        right_child_name = data['right_child_name']
        
        left_child = cls.from_dict_custom(data['left_child']) if data['left_child'] is not None else None
        right_child = cls.from_dict_custom(data['right_child']) if data['right_child'] is not None else None
        children = [child for child in [left_child, right_child] if child is not None]
        
        return cls(
            module_id=module_id,
            module_name=module_name,
            features=features,
            feature_names=feature_names,
            output=output,
            total_output=total_output,
            children=children,
        )
    @classmethod
    def from_dict(cls, data):
        return cls(
            module_id=data['module_id'],
            module_name=data['module_name'],
            features=data['features'],
            feature_names=data['feature_names'],
            output=data['output'],
            total_output=data['total_output'],
            children=[cls.from_dict(child) for child in data['children']] if data['children'] else [],
        
        )
    
class QueryPlanNode(Node):
    def __init__(
        self, 
        module_id,
        module_name,
        features,
        feature_names,
        output,
        total_output,
        children,
        input_output_features_schema=None,
        ):
        super().__init__(module_id, module_name, features, feature_names, output, total_output, children)
        self.left_child = self.children[0] if len(children) > 0 else None
        self.right_child = self.children[1] if len(children) > 1 else None
        self.left_child_name = self.left_child.module_name if self.left_child else None
        self.right_child_name = self.right_child.module_name if self.right_child else None
        self.input_output_features_schema = input_output_features_schema
        self.three_children = False
        self.input_output_features_schema = input_output_features_schema
        

    def to_dict(self,input_output_features_schema):
        node_dict = {
            "module_id": self.module_id,
            "module_name": self.module_name,
            "features": self.features,
            "feature_names": sorted(input_output_features_schema.get(self.module_name, None)["input_features"]),
            "output": self.output,
            "total_output": self.total_output,
            "left_child_name": self.left_child_name,
            "right_child_name": self.right_child_name,
        }
        if self.children:
            node_dict["children"] = [child.to_dict(input_output_features_schema) for child in self.children]
        else:
            node_dict["children"] = None
        return node_dict

    def return_outputs_as_list_postorder(self, outcomes_parallel=False):
        outputs = []
        
        if self.left_child:
            outputs.extend(self.left_child.return_outputs_as_list_postorder(outcomes_parallel=outcomes_parallel))
        
        if self.right_child:
            outputs.extend(self.right_child.return_outputs_as_list_postorder(outcomes_parallel=outcomes_parallel))
        
        if outcomes_parallel:
            if self.output_run_time is not None:
                outputs.append(self.output_run_time)
        else:
            if self.output_total_run_time is not None:
                outputs.append(self.output_total_run_time)
        
        return outputs

    def return_outputs_as_list_preorder(self, outcomes_parallel=False):
        outputs = []
        
        if outcomes_parallel:
            if self.output is not None:
                outputs.append(self.output)
        else:
            if self.total_output is not None:
                outputs.append(self.total_output)
        
        if self.left_child:
            outputs.extend(self.left_child.return_outputs_as_list_preorder(outcomes_parallel=outcomes_parallel))
        
        if self.right_child:
            outputs.extend(self.right_child.return_outputs_as_list_preorder(outcomes_parallel=outcomes_parallel))
        
        return outputs

    def get_depth(self):
        if self.left_child is None and self.right_child is None:
            return 1
        left_depth = self.left_child.get_depth() if self.left_child else 0
        right_depth = self.right_child.get_depth() if self.right_child else 0
        return max(left_depth, right_depth) + 1

    
    @classmethod
    def from_dict_custom(cls, data, input_output_features_schema):
        module_name = data['Node Type']
        module_id = data['Node Type']
        node_input_feature_names = input_output_features_schema.get(module_name, None)["input_features"]
        # sort the features by name
        node_input_feature_names = sorted(node_input_feature_names)
        # print(module_id, data.keys())
        # if feature is a list then print feature name and skip
        features = []
        for feature_name in node_input_feature_names:
            if feature_name in data:
                if isinstance(data[feature_name], list):
                    # print(feature_name, module_name)
                    features.append(data.get(feature_name, None))
            
            features.append(data.get(feature_name, None))
        output = data['Self Time']
        total_output = data['Actual Total Time']
        children = []
        if "Plans" in data:
            for count, plan in enumerate(data['Plans']):
                
                if count == 0:
                    left_child = cls.from_dict_custom(plan, input_output_features_schema)
                    children.append(left_child)
                    left_child_name = left_child.module_name
                elif count == 1:
                    right_child = cls.from_dict_custom(plan, input_output_features_schema)
                    children.append(right_child)
                    right_child_name = right_child.module_name
                else:
                    # print("Three children found for a node.")
                    cls.three_children = True
                    another_child = cls.from_dict_custom(plan, input_output_features_schema)
                    children.append(another_child)
        else:
            left_child_name = None
            right_child_name = None
            left_child = None
            right_child = None
        return cls(
            module_id=module_id,
            module_name=module_name,
            features=features,
            feature_names=node_input_feature_names,
            output=output,
            total_output=total_output,
            children=children,
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            module_id=data['module_id'],
            module_name=data['module_name'],
            features=data['features'],
            feature_names=data['feature_names'],
            output=data['output'],
            total_output=data['total_output'],
            children=[cls.from_dict(child) for child in data['children']] if data['children'] else [],
        
        )

        



