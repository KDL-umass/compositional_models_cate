import psycopg2
import json
import os
import re
import logging
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def start_conn(config, timeout = 3600000):
    conn = psycopg2.connect(dbname=config["dbname"], 
                            user=config["user"], 
                            password=config["password"], 
                            host=config["host"], 
                            port=config["port"], options='-c statement_timeout={}'.format(timeout))
    
    return conn

def generate_basic_config(config_file_path, dbname = "stackoverflow2014", data_folder_name = "data_v2"):
    config_dict = {}
    config_dict["dbname"] = dbname
    config_dict["user"] = "postgres"
    config_dict["password"] = ""
    config_dict["host"] = "localhost"
    config_dict["port"] = "5432"
    config_dict["postgres_dir"] = "/usr/local/var/postgres"
    config_dict["main_dir"] = "{}/queries".format(ROOT_DIR)
    config_dict["data_folder_name"] = "{}/data/{}".format(config_dict["main_dir"], data_folder_name)
    config_dict["user_queries_dir"] = "{}/user_queries".format(config_dict["data_folder_name"])
    config_dict["rewritten_queries_dir"] = "{}/rewritten_queries".format(config_dict["data_folder_name"])
    config_dict["parameterized_queries_dir"] = "{}/parameterized_queries".format(config_dict["data_folder_name"])
    config_dict["processed_queries_dir"] = "{}/processed_queries".format(config_dict["data_folder_name"])
    config_dict["executed_queries_dir"] = "{}/executed_queries".format(config_dict["data_folder_name"])
    config_dict["post_processed_queries_dir"] = "{}/post_processed_queries".format(config_dict["data_folder_name"])
    config_dict["csvs_dir"] = "{}/all_csvs".format(config_dict["data_folder_name"])
    config_dict["logs_dir"] = "{}/logs".format(config_dict["data_folder_name"])
    # generate_dirs(config_dict)
    with open(config_file_path, "w", encoding='utf-8') as fp:
        json.dump(config_dict, fp, ensure_ascii=False, indent=4)

    return config_dict

def generate_dirs(config):
    if not os.path.exists(config["main_dir"]):
        os.makedirs(config["main_dir"])

    if not os.path.exists(config["logs_dir"]):
        os.makedirs(config["logs_dir"])

    if not os.path.exists(config["user_queries_dir"]):
        os.makedirs(config["user_queries_dir"])

    if not os.path.exists(config["rewritten_queries_dir"]):
        os.makedirs(config["rewritten_queries_dir"])

    if not os.path.exists(config["parameterized_queries_dir"]):
        os.makedirs(config["parameterized_queries_dir"])

    if not os.path.exists(config["executed_queries_dir"]):
        os.makedirs(config["executed_queries_dir"])

    if not os.path.exists(config["processed_queries_dir"]):
        os.makedirs(config["processed_queries_dir"])

    if not os.path.exists(config["post_processed_queries_dir"]):
        os.makedirs(config["post_processed_queries_dir"])

    if not os.path.exists(config["csvs_dir"]):
        os.makedirs(config["csvs_dir"])

def return_query_id_from_file(file_name):
    match = re.search(r'query_(\w+)', file_name) 
    if match:                                    # Check if there is a match first
        return int(match.group(1))
    else:
        return -1

def return_query_id_from_link(link):
    match = re.search(r'/query/(\w+)', link) 
    if match:                                    # Check if there is a match first
        return int(match.group(1))
    else:
        return -1

def set_logging(logname):
    logging.basicConfig(filename=logname,
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

def get_indiv_treatement_idx(t_idx):
#     page_cost_level = t_idx/9`
#     t_idx = t_idx % 9
#     memory_level = t_idx / 3
#     index_level = t_idx % 3
    
    index_level = t_idx/9
    t_idx = t_idx % 9
    memory_level = t_idx / 3
    page_cost_level = t_idx % 3
    return int(index_level), int(memory_level), int(page_cost_level)

def get_treatment_idx(index_level, memory_level, page_cost_level):
#     return int(index_level + memory_level * 3 + page_cost_level * 3 * 3)
    return int(page_cost_level + memory_level * 3 + index_level * 3 * 3)




