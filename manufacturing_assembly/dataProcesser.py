import os
import json
import csv
from collections import defaultdict
import pandas as pd
from copy import deepcopy

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def read_file_high_level (initial_conditions, simulation_result):
    query_id = initial_conditions.split('/')[-1]
    query_id = "{}_{}".format(query_id.split('_')[1], query_id.split('_')[3])
    query_id = query_id.split('.')[0]
    initial_conditions = load_json(initial_conditions)
    simulation_result = load_json(simulation_result)
    row = { 'query_id' : query_id,
            'treatment_id': simulation_result["treatment_id"],
            "raw_material": 0,
            "electronic_component": 0,
            "misc_component": 0,
            "fastener": 0,
            'total_output': simulation_result["query_output"]['total_output'],
            'total_time': simulation_result["query_output"]['factory_run_time'],
            'tree_depth': 0
            }
    layer_num = 0

    for layer in initial_conditions['assembly_sequence']:
        layer_num +=1
        for process in layer:
            name_layer = "{}_layer_{}".format(process['name'], layer_num)
            row[name_layer] = row[name_layer] + 1 if name_layer in row else 1
            for key in process['inventory_input_items'].keys():
                row[key] += process['inventory_input_items'][key]
                row['tree_depth'] = layer_num

    return row

def read_file_low_level(simulation_result_dir):
    scenario_id = simulation_result_dir.split('/')[-1].split('_')
    scenario_demand = f"{scenario_id[1]}_{scenario_id[3]}"
    results = load_json(simulation_result_dir)
    treatment_id = results['treatment_id']
    rows = []

    def process_node(node, parent_id=None):
        row = {
            'query_id': scenario_demand,
            'treatment_id': treatment_id,
            'process': node['module_name'],
            'process_id': node['module_id'],
            'raw_material': 0,
            'electronic_component': 0,
            'misc_component': 0,
            'fastener': 0,
            'material-process_part': 0,
            'material-join_part': 0,
            'electronics_part': 0,
            'assembly_part': 0,
            'process_runtime': node['output']['station_Active_time'],
            'parts_produced_by_process': node['output']['Total_WIPs_produced']
        }

        for i, value in enumerate(node['features']):
            row[node['feature_names'][i]] = value

        rows.append(row)

        for child in node.get('children', []):
            process_node(child, node['module_id'])

    process_node(results['query_output']['json_tree'])
    return rows

def dicts_to_dataframe(dict_list):
    # Get all unique keys from all dictionaries
    all_keys = set().union(*dict_list)
    
    # Create a list of dictionaries with all keys, filling missing values with 0
    normalized_dicts = [{key: d.get(key, 0) for key in all_keys} for d in dict_list]
    
    # Create DataFrame from the normalized list of dictionaries
    return pd.DataFrame(normalized_dicts)