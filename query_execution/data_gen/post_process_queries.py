import json
import time
import argparse
import os
from unittest import result
import psycopg2
import psycopg2.extras 
from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE
from os import listdir
from select import select
from os.path import isfile, join
import logging 
# from treatment_implementation import set_treatment_config, get_treatment_config, get_postgres_status, rstart_postgres, stop_postgres, restart_postgres
from treatment_implementation import *
import time
from data_utils import  *
import numpy as np
import pandas as pd
import tqdm
import math
import warnings
from plan_tree import *

def store_post_processed_query(file_name, post_processed_dict, query_info):
    final_dict = {}
    for key, val in query_info.items():
        if key != "json_result":
            final_dict[key] = val

    final_dict["json_result"] = [{"Plan": post_processed_dict, "Execution Time": query_info["json_result"][0]["Execution Time"], "Planning Time": query_info["json_result"][0]["Planning Time"]}]
    with open(file_name, "w") as f:
        json.dump(final_dict, f, indent=4)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# arguments contain number of queries to execute and intervention settings
parser = argparse.ArgumentParser()
# parser.add_argument('--N', type = int, default = 0, help = "number of queries to process")

parser.add_argument('--dbname', type = str, required = True, help = "database name")
parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
parser.add_argument('--rerun', type = int, default = 0, help = "if query is re-procesed")
parser.add_argument('--query_template', type = int, default = 0, help = "query_template to be run ")
parser.add_argument('--query_param', type = int, default = 0, help = "query_paramterization to be run ")
parser.add_argument('--index_type', type = str, default = "hash", help = "index type to be used")
parser.add_argument('--disable_parallelization', type = int, default = 1, help = "should parallelism be disabled")
parser.add_argument('--enable_indexscan', type = int, default = 1, help = "should indexscan be enabled")

if __name__ == '__main__':
    # load arguments
    args = parser.parse_args()
    # N = args.N
    
    dbname = args.dbname
    data_folder_name = args.data_folder_name
    rerun = args.rerun
    disable_parallelization = args.disable_parallelization
    index_type = args.index_type
    query_template = args.query_template
    enable_indexscan = args.enable_indexscan
    if query_template < 10:
        query_template = "0{}".format(query_template)
    
    query_param = args.query_param
    # separate_tpch = args.separate_tpch
   

    # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = "{}/{}".format(config_path, config_name)
    config = generate_basic_config(config_file_path, args.dbname, args.data_folder_name)

    # print config
    print(config)
   
    with open('{}/operations_schema_all_features.json'.format(config_path)) as f:
        operations_all_features_schema = json.load(f)
    
    with open('{}/operations_schema_input_output_features.json'.format(config_path)) as f:
        operations_input_output_schema = json.load(f)

    executed_queries_dir_base = config["executed_queries_dir"]
    post_processed_dir_base = config["post_processed_queries_dir"]
    csvs_dir = config["csvs_dir"]
    
    if args.dbname == "tpch" and "tpch_scans" not in args.data_folder_name:
        
        executed_queries_dir_base = "{}/{}/".format(executed_queries_dir_base, query_template)
        post_processed_dir_base = "{}/{}/".format(post_processed_dir_base, query_template)
        csvs_dir = "{}/{}/".format(csvs_dir, query_template)

    all_folders = [f for f in listdir(executed_queries_dir_base) if not isfile(join(executed_queries_dir_base, f))]
    positive_errors = 0
    positive_errors_list = []
    total_errors = 0
    total_errors_list = []
    cte_queries = []
    all_dicts = {}
    # all_dfs = {}
    
    for fold in all_folders:
        all_new_jsons = {}
        try:
            index_level = int(fold.split("_")[1])
        except:
            continue
        memory_level = int(fold.split("_")[3])
        page_cost_level = int(fold.split("_")[5])
        tid = get_treatment_idx(index_level, memory_level, page_cost_level)
        executed_queries_dir = "{}/{}".format(executed_queries_dir_base, fold)
        post_processed_dir = "{}/{}".format(post_processed_dir_base, fold)
        if not os.path.exists(post_processed_dir):
            os.makedirs(post_processed_dir)
        all_files = [f for f in listdir(executed_queries_dir) if isfile(join(executed_queries_dir, f))]
        all_files.sort(key=lambda x: return_query_id_from_file(x))
        # N = args.N
        # start_index = args.start_index
        # if N == 0:
        #     all_files = all_files[start_index:]
        # else:
        #     all_files = all_files[start_index:start_index+N]

        print("Total number of files in {} dir {}".format(executed_queries_dir, len(all_files)))
    
    
        # print(ops_schema)
        # instantiate the query plan structure and compare the time from the components, this should work for all the queries 
        
        
        count = 0
        
        for i in tqdm.tqdm(range(len(all_files))):
            file = all_files[i]
           
            qid = file.split("_")[2]

            rid = file.split("_")[-1].split(".")[0]
            # print(qid, tid, rid)
            file_read_path = "{}/{}".format(executed_queries_dir, file)
            file_write_name = os.path.join(post_processed_dir, "postgres_query_" + str(qid) + "_" + str(tid) + "_" + str(rid) + ".json")
            if os.path.exists(file_write_name) and not rerun:
                continue
            with open(file_read_path, 'r') as f:
                query = json.load(f)
            specific_json = query["json_result"][0]
            # if args.dbname == "tpch":
            #     if int(query["query_template"]) < 10:
            #         qid = "0{}".format(query["query_template"]) + str(query["query_paramid"])
            #         file_write_name = os.path.join(post_processed_dir, "postgres_query_" + str(qid) + "_" + str(tid) + "_" + str(rid) + ".json")
            #         query["query_id"] = qid
            plan = specific_json["Plan"]
            execution_time = specific_json["Execution Time"]
            planning_time = specific_json["Planning Time"]
            plan_explainer = get_plan_explainer(qid, tid, rid, plan)
                    # global op 
                    # op = []
                    # parse_query_plan(plan,  count = 0, ops_schema=ops_schema, ops=op)
            total_time = plan_explainer.get_total_time(plan_explainer.plan_tree)
            negative_check = plan_explainer.check_negative_time(plan_explainer.plan_tree)

                    
            if negative_check:
                print("Negative time for ", qid, tid, rid)
                plan_explainer.print_tree(plan_explainer.plan_tree)
                positive_errors += 1
                positive_errors_list.append((qid, tid))
                
            if not math.isclose(total_time, plan["Actual Total Time"], rel_tol=1e-04, abs_tol=0.0):
                print("Calculated Total time: ", total_time, "Actual Total Time: ", plan["Actual Total Time"])
                print("Check the log time", qid, tid, rid)
                plan_explainer.print_tree(plan_explainer.plan_tree)
                total_errors += 1
                total_errors_list.append((qid, tid))
            # plan_explainer.get_df_per_operation(plan_explainer.plan_tree)
            plan_explainer.get_training_data(plan_explainer.plan_tree, operations_input_output_schema)
            post_processed_dict = plan_explainer.convert_tree_to_dict(plan_explainer.plan_tree)
            if qid not in all_new_jsons:
                all_new_jsons[qid] = {}
            if tid not in all_new_jsons[qid]:
                all_new_jsons[qid][tid] = {}
            all_new_jsons[qid][tid][rid] = {"dict": post_processed_dict, "query_info": query}
            for k,v in plan_explainer.training_data.items():
                if k not in all_dicts:
                    all_dicts[k] = []
                # don't create list of lists, create list of dicts
                for row in v:
                    if dbname == "tpch":
                        row["query_paramid"] = query["query_paramid"]
                        row["query_template"] = query["query_template"]
                    all_dicts[k].append(row)
            # store_post_processed_query(qid, tid, rid, post_processed_dict)

            # for key in plan_explainer.all_dfs.keys():
            #     if key in all_dfs:
            #         all_dfs[key] = np.concatenate([all_dfs[key], plan_explainer.all_dfs[key]])
            #     else:
            #         all_dfs[key] = plan_explainer.all_dfs[key]
            count += 1

        for qid, value in all_new_jsons.items():
            for tid, value in value.items():
                for rid, value in value.items():
                    file_write_name_final = os.path.join(post_processed_dir, "postgres_query_" + str(qid) + "_" + str(tid) + "_" + str(rid) + ".json")
                    store_post_processed_query(file_write_name_final, value["dict"], value["query_info"])
          
    
    if not os.path.exists(csvs_dir):
        os.makedirs(csvs_dir)
    print(all_dicts.keys())
    for k in all_dicts.keys():
        # print(all_dicts[k])
        ops_df = pd.DataFrame(all_dicts[k])
        ops_df["query_id"] = ops_df["query_id"].astype(str)
        print(ops_df["treatment_id"].value_counts())
        file_name = "{}/{}.csv".format(csvs_dir, k)
        print("Writing to file ", file_name)
        ops_df.to_csv(file_name, index=False)

    # for key in all_dfs.keys():
    #     # print(all_dfs[key])
    #     # print(operations_schema[key])
    #     ops_df = pd.DataFrame(all_dfs[key], columns = ["query_id", "treatment_id", "run_id"] + operations_schema[key])
    #     ops_df.to_csv("{}/{}.csv".format(csvs_dir,key))
    #     # print(key, len(all_dfs[key]))

    
                    
                    
                
        