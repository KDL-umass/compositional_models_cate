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

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore")
# create arguments for the SeqScan using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type = str, required = True, help = "database name")
parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
parser.add_argument('--query_template', type = int, default = 0, help = "query_template to be run ")
parser.add_argument('--query_param', type = int, default = 0, help = "query_paramterization to be run ")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

features = ["query_id", "treatment_index", "run_id", "num_ref_tables", "total_ref_rows", "avg_num_rows",
            "avg_num_cols", "max_num_rows", "max_num_cols", "len_chars", "num_joins", "num_group_by", 
            "num_orders", "index_level", "memory_level", "page_cost_level", "num_rows", "plan_total_cost", "plan_start_cost", "plan_rows", "plan_width",
            "limit_present", "parameters_present", "execution_time", "adjusted_execution_time", 
            "shared_hit_blocks", "shared_read_blocks", "temp_read_blocks", "temp_written_blocks"]
avail_features = ["num_ref_tables", "total_ref_rows", "avg_num_rows",
            "avg_num_cols", "max_num_rows", "max_num_cols", "len_chars", "num_joins", "num_group_by", 
            "num_orders", "index_level", "memory_level", "page_cost_level"]
exclusion_features = ["query_id", "treatment_index", "num_rows", "run_id", "execution_time", "execution_time_log", "adjusted_execution_time", "adjusted_execution_time_log", "shared_hit_blocks", "shared_read_blocks", "temp_read_blocks", "temp_written_blocks"]

covariates = [ "avg_num_rows", "avg_num_cols", "plan_total_cost", "plan_start_cost", "plan_rows", "plan_width", "len_chars", "num_ref_tables", "num_joins", "num_group_by", "num_orders", "limit_present", "parameters_present"]
treatments = ["index_level", "memory_level", "page_cost_level"]
outcomes = ["adjusted_execution_time_log"]

if __name__ == '__main__':
    args = parser.parse_args()
    # N = args.N
    
    dbname = args.dbname
    data_folder_name = args.data_folder_name
    query_template = args.query_template
    if query_template < 10:
        query_template = "0{}".format(query_template)
    
    query_param = args.query_param
    

    # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(ROOT_DIR)
    config_file_path = "{}/{}".format(config_path, config_name)
    config = generate_basic_config(config_file_path, args.dbname, args.data_folder_name)


    post_processed_dir_base = config["post_processed_queries_dir"]
    csvs_dir = config["csvs_dir"]

    if args.dbname == "tpch":
        post_processed_dir_base = "{}/{}/".format(post_processed_dir_base, query_template)
        csvs_dir = "{}/{}/".format(csvs_dir, query_template)
        # if separate_tpch:
        #     post_processed_dir_base = "{}/{}/".format(post_processed_dir_base, query_param)
        #     csvs_dir = "{}/{}/".format(csvs_dir, query_param)


    if os.path.exists(post_processed_dir_base):
        all_folders = listdir(post_processed_dir_base)
        all_folders = [f for f in listdir(post_processed_dir_base) if not isfile(join(post_processed_dir_base, f))]
        
        data_path = "{}/{}".format(csvs_dir, "all_data.csv")
        all_rows = []
        jsons = {}
        for fold in all_folders:
            index_level = int(fold.split("_")[1])
            memory_level = int(fold.split("_")[3])
            page_cost_level = int(fold.split("_")[5])
            tid = get_treatment_idx(index_level, memory_level, page_cost_level)
            post_processed_dir = "{}/{}".format(post_processed_dir_base, fold)
            count = 0
            all_files = [f for f in listdir(post_processed_dir) if isfile(join(post_processed_dir, f))]
            all_files.sort(key=lambda x: return_query_id_from_file(x))
            # N = args.N
            # start_index = args.start_index
            # if N == 0:
            #     all_files = all_files[start_index:]
            # else:
            #     all_files = all_files[start_index:start_index+N]
            print("Number of files: {} in {}".format(len(all_files), post_processed_dir))
            print(len(all_rows))
            for i in tqdm.tqdm(range(len(all_files))):
                file = all_files[i]
                qid = file.split("_")[2]
                rid = int(file.split("_")[-1].split(".")[0])
                file_name = os.path.join(post_processed_dir, "postgres_query_" + str(qid) + "_" + str(tid) + "_" + str(rid) + ".json")
            
                with open(file_name, 'r') as f:
                    query = json.load(f)
                if qid not in jsons.keys():
                    jsons[qid] = {}
                row = []
                row.append(qid)
                row.append(int(tid))
                row.append(rid)
                for feat in avail_features:
                    if feat in query.keys():
                        row.append(query[feat])
                    else:
                        row.append(None)
                jsons[qid]["query"] = query["query_text"]
                jsons[qid]["url"] = query["location"]
                if tid not in jsons[qid].keys():
                    jsons[qid][tid] = {}
                jsons[qid][tid][rid] = query["json_result"]
                num_rows = query["json_result"][0]["Plan"]["Actual Rows"]
                plan_tot_cost = query["json_result"][0]["Plan"]["Total Cost"]
                plan_start_cost = query["json_result"][0]["Plan"]["Startup Cost"]
                plan_rows = query["json_result"][0]["Plan"]["Plan Rows"]
                plan_width = query["json_result"][0]["Plan"]["Plan Width"]
                row.append(num_rows)
                row.append(plan_tot_cost)
                row.append(plan_start_cost)
                row.append(plan_rows)
                row.append(plan_width)
                if 'limit' in query["query_text"]:
                    row.append(True)
                else:
                    row.append(False)
                if len(query["parameters"]) > 0:
                    row.append(True)
                else:
                    row.append(False)
                row.append(query["json_result"][0]["Plan"]['Actual Total Time'])
                row.append(query["json_result"][0]["Plan"]['Total Time'])
                row.append(query["json_result"][0]["Plan"]["Shared Hit Blocks"])
                row.append(query["json_result"][0]["Plan"]["Shared Read Blocks"])
                row.append(query["json_result"][0]["Plan"]["Temp Read Blocks"])
                row.append(query["json_result"][0]["Plan"]["Temp Written Blocks"])
                if args.dbname == "tpch":
                    row.append(query["query_template"])
                    row.append(query["query_paramid"])
                    
                all_rows.append(row)
                    
        if args.dbname == "tpch":
            features = features + ["query_template", "query_paramid"]
        df = pd.DataFrame(all_rows, columns = features)
        print(df.columns)
        df["execution_time_log"] = np.log10(df["execution_time"])
        df["adjusted_execution_time_log"] = np.log10(df["adjusted_execution_time"])
        print("Number of features: {}".format(len(features)))
        print("Number of rows: {}".format(len(df)))

        # take median of numeric columns and unique of non-numeric columns

        # numeric_columns = list(set(list(df._get_numeric_data().columns) + ["query_id", "treatment_index"]))
        # non_numeric_columns = list(set(list(set(df.columns) - set(numeric_columns)) + ["query_id", "treatment_index"]))
        # # aggregate non-numeric columns using unique
        # non_numeric_df = df[non_numeric_columns].groupby(["query_id", "treatment_index"]).unique().reset_index()
        # aggregate numeric columns using median
        # numeric_df = df[numeric_columns].groupby(["query_id", "treatment_index"]).median().reset_index()
        # merge numeric and non-numeric columns
        # data = pd.merge(numeric_df, non_numeric_df, on=["query_id", "treatment_index"])
        if args.dbname == "tpch":
            data = df.groupby(["query_id", "treatment_index", "query_template", "query_paramid"]).median().reset_index()
        else:
            data = df.groupby(["query_id", "treatment_index"]).median().reset_index()
        print("Number of rows after grouping: {}".format(len(data)))
        all_query_ids = list(set(data["query_id"].values))
        print(data.columns)
        # print(all_query_ids)
        # print(jsons.keys())

            
        # get query plan features
        
        all_ops = {}
        query_operations = []
        query_relations = []
        query_indices = []
        parallel_aware = []
        for qid in all_query_ids:
            all_ops[qid] = {}
            for tid in list(jsons[qid].keys()):
                if tid in ["query", "url"]:
                    continue
                
                rid = 1
                if rid not in jsons[qid][tid].keys():
                    rid = 0
                
                # print(qid, tid, rid)
                # print(jsons[qid][tid][rid])
                plan = jsons[qid][tid][rid][0]["Plan"]
                
                plan_explainer = PlanExplainer(qid, tid, rid, plan, set_total_time=False)
                plan_explainer.get_all_operations(node=plan_explainer.plan_tree)
                all_ops[qid][tid] = plan_explainer.all_operations
                for ops in plan_explainer.all_operations:
                    op = ops[0]
                    actual_rows = ops[1]
                    plan_rows = ops[2]
                    bytes = ops[3]
                    rel = ops[4]
                    idx = ops[5]
                    parallel_aware = ops[6]
                    
                    query_operations.append(op)
                    query_relations.append(rel)
                    query_indices.append(idx)
                    

        query_operations = list(set(query_operations))
        query_relations = list(set(query_relations))
        if None in query_operations:
            query_relations.remove(None)

        query_indices = list(set(query_indices))
        if None in query_indices:
            query_indices.remove(None)
        query_operation_actual_rows = []
        query_operation_plan_rows = []
        for q in query_operations:
            print(q)
            query_operation_actual_rows.append(q + "_actual_rows")
            query_operation_plan_rows.append(q + "_plan_rows")
        plan_features = query_operations + ["parallel_aware"] + ["total_actual_row_count", "total_plan_row_count", "total_byte_count"]  + query_operation_actual_rows +  query_operation_plan_rows + query_relations + query_indices
        print("Number of plan features: {}".format(len(plan_features)))
        print(plan_features)


        new_rows = []
        for index,row in data.iterrows():
            qid = row["query_id"]
            tid = int(row["treatment_index"])
            plan = all_ops[qid][tid]
            ops_row = np.zeros(len(query_operations) + 1)
            ops_actual_row_count = np.zeros(len(query_operations))
            ops_plan_row_count = np.zeros(len(query_operations))
            rels_row = np.zeros(len(query_relations))
            idxs_row = np.zeros(len(query_indices))
            total_actual_row_count = 0
            total_plan_row_count = 0
            total_byte_count = 0
            for ops in plan:
                op = ops[0]
                actual_rows = ops[1]
                plan_rows = ops[2]
                bytes = ops[3]
                rel = ops[4]
                idx = ops[5]
                parallel_aware = ops[6]
                    
                parallel_aware = ops[5]
                total_actual_row_count += actual_rows
                total_plan_row_count += plan_rows
                total_byte_count += bytes
                if op is not None:
                    ops_row[query_operations.index(op)] += 1
                    ops_actual_row_count[query_operations.index(op)] += actual_rows
                    ops_plan_row_count[query_operations.index(op)] += plan_rows
                if rel is not None:
                    rels_row[query_relations.index(rel)] += 1
                if idx is not None:
                    idxs_row[query_indices.index(idx)] += 1

                if parallel_aware:
                    ops_row[-1] += 1

            new_row = np.concatenate([[qid],[int(tid)],ops_row, [total_actual_row_count, total_plan_row_count, total_byte_count], ops_actual_row_count, ops_plan_row_count, rels_row, idxs_row])
            new_rows.append(new_row)
        new_df = pd.DataFrame(new_rows, columns = ["query_id", "treatment_index"] + plan_features)
        new_df["treatment_index"] = new_df["treatment_index"].astype(int)
        # print data type of each column
        # print(new_df.dtypes)
        # print(data.dtypes)
        data = pd.merge(data, new_df, on=["query_id", "treatment_index"])
        
            
        if not os.path.exists(data_path):
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # save data in all_csvs
        data.to_csv(data_path, index=False, float_format='%.3f')
        

        