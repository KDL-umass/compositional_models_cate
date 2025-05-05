from os import listdir
from os.path import isfile, join
import re
import json
import argparse
import psycopg2
import random
import os 
from data_utils import  start_conn, generate_basic_config
from treatment_implementation import get_postgres_status, start_postgres


PARAM_RE = re.compile("%\((?P<paramname>[A-z]+)\)s")
def main():
    # for now we are storing the same user queries used by previous NeurIPS and ICML papers. 
    # This can change if needed 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dbname', type = str, required = True, help = "database name")
    parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
    
    args = parser.parse_args()
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(ROOT_DIR)
    config_file_path = "{}/{}".format(config_path, config_name)

    
    config = generate_basic_config(config_file_path, args.dbname, args.data_folder_name)
    
    print(config)
    status = get_postgres_status(config)
    if status == False:
        start_postgres(config)
    rewritten_queries_dir = config["rewritten_queries_dir"]
    parameterized_queries_dir = config["parameterized_queries_dir"]
    if not os.path.exists(parameterized_queries_dir):
        os.makedirs(parameterized_queries_dir)

    read_path = rewritten_queries_dir
    all_files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    print("Total number of files in {} dir {}".format(read_path, len(all_files)))
    write_path = parameterized_queries_dir
    conn = start_conn(config)
    cursor = conn.cursor()
    # TO - DO store all user IDs and tags beforehand and select randomly from all rather than top K otherwise 
    # paremeterized queries might be biased towards certain users only. 
    cursor.execute("select Id from users order by reputation")
    user_ids = [u[0] for u in cursor.fetchall()]
    print(len(user_ids))
    # cursor.execute("select TagName from Tags order by \"count\"")
    # tag_names = [t[0] for t in cursor.fetchall()]
    # print(len(tag_names))
    cursor.execute("select DISTINCT DisplayName from users ")
    display_names = [t[0] for t in cursor.fetchall()]
    print(len(display_names))
    
    
    for file in all_files:
        read_file_name = "{}/{}".format(read_path, file)
        write_file_name = "{}/{}".format(write_path, file)
        if not os.path.isfile(write_file_name):
            with open(read_file_name, 'r') as f:
                query = json.load(f)
            query["parameters"] = dict()
            matches = set(PARAM_RE.findall(query["query_text"]))
            for match in matches:
                print(match, file)
                if match.lower() == "userid":
                    query["parameters"][match] = random.choice(user_ids)
                # elif match.lower() == "tagname":
                #     query["parameters"][match] = random.choice(tag_names)
                elif match.lower() == "name":
                    query["parameters"][match] = random.choice(display_names)
                else:
                    print("match not implemented")
                
            with open(write_file_name, "w", encoding='utf-8') as fp:
                json.dump(query, fp, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()





