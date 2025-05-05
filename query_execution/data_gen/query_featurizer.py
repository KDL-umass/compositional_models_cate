from email.policy import default
import psycopg2
import psycopg2.extensions
from os import listdir
from os.path import isfile, join
from dateutil import parser as dateparser
import re
import json
import argparse
import random
from collections import defaultdict
import os
from data_utils import  start_conn, generate_basic_config

missing_tables = {"stackoverflow2014": ['suggestededits', 'posthistory', 'suggestededitvotes', 'postfeedback', 'tagsynonyms'], 
                  "mathso": ['suggestededits', 'postfeedback', 'suggestededitvotes', 'tagsynonyms'],
                  "tpch": [],
                  "tpcds": [],
                  "tpch_10": []}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', type = str, default = "mathso", help = "database name")
    parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
    parser.add_argument('--query_template', type = int, default = 0, help = "query_template to be run ")

    
    args = parser.parse_args()
    query_template = args.query_template
    if query_template < 10:
        query_template = "0{}".format(query_template)
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
     # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = "{}/{}".format(config_path, config_name)
    config = generate_basic_config(config_file_path, args.dbname, data_folder_name=args.data_folder_name)
    
    if args.dbname == "mathso":
        if args.data_folder_name not in ["mathso_scans", "mathso_joins_scm", "mathso_scans_scm"]:
            read_path = config["parameterized_queries_dir"]
            write_path = config["processed_queries_dir"]
        else:
            read_path = config["processed_queries_dir"]
            write_path = config["processed_queries_dir"]
    else:
        if "tpch_scans" not in args.data_folder_name:
            read_path = "{}/{}".format(config["parameterized_queries_dir"], query_template)
            write_path = "{}/{}".format(config["processed_queries_dir"], query_template)
        else:
            read_path = config["processed_queries_dir"]
            write_path = config["processed_queries_dir"]

    if not os.path.exists(write_path):
        os.makedirs(write_path)
    
    all_files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    print("Total number of files in {} dir {}".format(read_path, len(all_files)))

    # read data info file 
    
    json_file_path = "{}/{}{}".format(config["data_folder_name"], config["dbname"], "_table_info.json")
    with open(json_file_path) as f:
        data_info = json.load(f)
    table_names = data_info.keys()
   
    

    if args.dbname in ["mathso"] and args.data_folder_name not in ["mathso_scans", "mathso_joins_scm", "mathso_scans_scm"]:
        queries_per_user = defaultdict(int)
        for file in all_files:
            file_path = "{}/{}".format(read_path, file)
            
            with open(file_path, 'r') as f:
                query = json.load(f)
        
            if query["user_href"] != None:
                queries_per_user[query["user_href"]] += 1

    for file in all_files:
        read_file_name = "{}/{}".format(read_path, file)
        write_file_name = "{}/{}".format(write_path, file)
        
        
        with open(read_file_name, 'r') as f:
            query = json.load(f)

            # remove punctuations and convert to lower text
            query_text = query["query_text"]
            query["failure"] = False
            new_text = re.sub('--.*\n', '', query_text)
            text_lower = new_text.lower()
            text_noPunct = re.sub('[,:]', ' ', new_text)
            text_lower_noPunct = re.sub('[,;]', ' ', text_lower)
            text_lower_noPunct = re.sub('\(', ' ( ', text_lower_noPunct)
            text_lower_noPunct = re.sub('\)', ' ) ', text_lower_noPunct)
            text_lower_noParen = re.sub('[\(\)]', '', text_lower_noPunct)

            words_lower = text_lower.split()
            words = new_text.split()
            words_noPunct = text_noPunct.split()
            words_lower_noPunct = text_lower_noPunct.split()

            order_by = 0
            group_by = 0
            for i in range(0, len(words_lower_noPunct)):
                if words_lower_noPunct[i] == "order" and words_lower_noPunct[i+1] == "by":
                        order_by += 1
                elif words_lower_noPunct[i] == "group" and words_lower_noPunct[i+1] == "by":
                        group_by += 1 

            # number of table references
            # uses distinct rather than double CHECK if need to refer the unique reference to the table.
            tables_used = set()
            all_tables_used = set()
            
    
            for i in range(0, len(words_lower_noPunct)):
                if words_lower_noPunct[i] in table_names:
                    tables_used.add(words_lower_noPunct[i])
                    all_tables_used.add(words_lower_noPunct[i])
                if words_lower_noPunct[i] in ["from", "join"]:
                    table_name = words_lower_noPunct[i+1]
                    if table_name in missing_tables[config["dbname"]]:
                        query["failure"] = True
                

            tables_used_num_rows = []
            tables_used_num_cols = []
            for table_name in tables_used:
                tables_used_num_rows.append(data_info[table_name]["num_rows"])
                tables_used_num_cols.append(data_info[table_name]["num_cols"])
                
            
            if len(tables_used) == 0:
                avg_num_rows = 0
                avg_num_cols = 0
                max_num_rows = 0
                max_num_cols = 0
                total_rows = 0
            else:
                total_rows = sum(tables_used_num_rows)
                avg_num_rows = sum(tables_used_num_rows)/float(len(tables_used_num_rows))
                avg_num_cols = sum(tables_used_num_cols)/float(len(tables_used_num_cols))
                max_num_rows = max(tables_used_num_rows)
                max_num_cols = max(tables_used_num_cols)

            query["num_ref_tables"] = len(tables_used)
            query["total_ref_rows"] = total_rows
            query["avg_num_rows"] = avg_num_rows
            query["avg_num_cols"] = avg_num_cols
            query["max_num_rows"] = max_num_rows
            query["max_num_cols"] = max_num_cols
            query["len_chars"] = len(new_text)
            query["num_joins"] = words_lower.count("join")
            query["num_group_by"] = group_by
            query["num_orders"] = order_by

            # investigate any relationship between query year and runtime
            if args.dbname == "mathso" and args.data_folder_name not in ["mathso_scans", "mathso_joins_scm", "mathso_scans_scm"]:
                query["creation_year"] = dateparser.parse(query["create_time"]).year 
                if query["user_href"] != None:
                    query["queries_by_user"] = queries_per_user[query["user_href"]]
                else:
                    query["queries_by_user"] = 0

            with open(write_file_name, "w", encoding='utf-8') as fp:
                json.dump(query, fp, ensure_ascii=False, indent=4)

    

if __name__ == "__main__":
    main()  
