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
from tqdm import tqdm
os.environ['PGOPTIONS'] = '-c statement_timeout=1000'
class QueryError(ValueError):
    pass

def main():
    # arguments contain number of queries to execute and intervention settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type = int, default = 0, help = "number of queries to execute")
    parser.add_argument('--index_level', type = int, default = 0, help = "sets index level")
    parser.add_argument('--memory_level', type = int, default = 0, help = "sets memory level")
    parser.add_argument('--page_cost_level', type = int, default = 0, help = "sets the level of page costs")
    parser.add_argument('--config', type = int, default = 0, help = "should config be refreshed")
    parser.add_argument('--dbname', type = str, required = True, help = "database name")
    parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
    parser.add_argument('--rerun_query', type = int, default = 0, help = "should already containing queries be re-run")
    parser.add_argument('--start_index', type = int, default = 0, help = "should already containing queries be re-run")
    parser.add_argument('--query_id', type = int, default = 0, help = "query_id to be run ")
    
    parser.add_argument('--server_restart', type = int, default = 0, help = "should postgres be restarted after every run")
    parser.add_argument('--disable_parallelization', type = int, default = 1, help = "should parallelism be disabled")
    parser.add_argument('--enable_indexscan', type = int, default = 1, help = "should index scan be enabled")
    parser.add_argument('--query_template', type = int, default = 0, help = "query_template to be run ")
    parser.add_argument('--query_param', type = int, default = 0, help = "query_paramterization to be run ")
    parser.add_argument('--set_analyze_true', type = int, default = 1, help = "if query is executed ")
    parser.add_argument('--timeout', type = int, default = 300000, help = "timeout for query execution")
    parser.add_argument('--rerun', type = int, default = 0, help = "if query is re-executed")
    parser.add_argument('--index_type', type = str, default = "hash", help = "index type to be used")
    # load arguments
    args = parser.parse_args()
    
    
    dbname = args.dbname
    index_level = args.index_level
    memory_level = args.memory_level
    page_cost_level = args.page_cost_level
    data_folder_name = args.data_folder_name
    disable_parallelization = args.disable_parallelization
    enable_indexscan = args.enable_indexscan
    index_type = args.index_type
    set_analyze_true = args.set_analyze_true
    query_template = args.query_template
    if query_template < 10:
        query_template = "0{}".format(query_template)
    

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(ROOT_DIR)
    config_file_path = "{}/{}".format(config_path, config_name)
    config = generate_basic_config(config_file_path, dbname, data_folder_name)

    # print config
    
    log_dir = config["logs_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    processed_queries_dir = config["processed_queries_dir"]
    # check folder name contains a string
   
    if args.dbname in ["tpch", "tpcds"] and "tpch_scans" not in args.data_folder_name:
        processed_queries_dir = "{}/{}/".format(processed_queries_dir, query_template)

    executed_queries_dir = config["executed_queries_dir"]
    if args.dbname in ["tpch", "tpcds"] and "tpch_scans" not in args.data_folder_name:
        executed_queries_dir = "{}/{}/".format(executed_queries_dir, query_template)
    if not os.path.exists(executed_queries_dir):
        os.makedirs(executed_queries_dir)
    
    psycopg2.extensions.set_wait_callback(wait_select_inter)

    all_files = [f for f in listdir(processed_queries_dir) if isfile(join(processed_queries_dir, f))]
    all_files.sort(key=lambda x: return_query_id_from_file(x))
    print("Total number of files in {} dir {}".format(processed_queries_dir, len(all_files)))
    
    print("Total number of selected files in {} dir {}".format(processed_queries_dir, len(all_files)))


    

    if args.query_id == 0:
        logname = "{}/execute_queries_ix_index_{}_memory_{}_page_{}.out".format(config["logs_dir"], args.index_level, args.memory_level, args.page_cost_level)
    else:
        logname = "{}/testing_query_id_ix_index_{}_memory_{}_page_{}_{}.out".format(config["logs_dir"], args.index_level, args.memory_level, args.page_cost_level, args.query_id)
    set_logging(logname)
    
    treatment_level_dir = "{}/index_{}_memory_{}_page_{}".format(executed_queries_dir, args.index_level, args.memory_level, args.page_cost_level)
    
    if not os.path.exists(treatment_level_dir):
        os.makedirs(treatment_level_dir)
    
    

    status = get_postgres_status(config)
    print("Postgres status: {}".format(status))
    if status == False:
        start_postgres(config)
    
    treatment_conf = get_treatment_config(config)
    print("Treatment config: {}".format(treatment_conf))
    
    set_index_level_flag = True
    set_memory_level_flag = True
    set_page_cost_level_flag = True
    if treatment_conf["index_level"] == index_level and treatment_conf["index_type"] == index_type:
        set_index_level_flag = False
    if treatment_conf["memory_level"] == memory_level:
        set_memory_level_flag = False
    if treatment_conf["page_cost"] == page_cost_level:
        set_page_cost_level_flag = False
    logging.info("Setting treatment level to index_{}_{}_memory_{}_{}_page_{}_{}".format(args.index_level, set_index_level_flag, args.memory_level, set_memory_level_flag, args.page_cost_level, set_page_cost_level_flag))
    set_treatment_config(config, index_level, memory_level, page_cost_level, disable_parallelization = disable_parallelization, set_index_level_flag = set_index_level_flag, set_memory_level_flag = set_memory_level_flag, set_page_cost_level_flag = set_page_cost_level_flag, enable_indexscan = enable_indexscan, index_type = index_type)
    treatment_conf = get_treatment_config(config)
    
    N = args.N
    start_index = args.start_index
    if N == 0:
        all_files = all_files[start_index:]
    else:
        all_files = all_files[start_index:start_index+N]
    
    failure_file_name = "{}/failures.txt".format(log_dir)
    failures = []
    # shuffle files inplace
    
    import random
    random.shuffle(all_files)
    print("Total number of files in {} dir {}".format(processed_queries_dir, len(all_files)))
    
    if args.query_id == 0:
        for count, file in enumerate(tqdm(all_files)):
            query_id = return_query_id_from_file(file)
            
            file_read_path = "{}/{}".format(processed_queries_dir, file)
            file_name = file.split(".")[0]
            run_id = 0
            file_write_path = "{}/{}_{}.json".format(treatment_level_dir, file_name, run_id)
            while os.path.isfile(file_write_path) and args.rerun == 1:
                run_id = run_id + 1
                file_write_path = "{}/{}_{}.json".format(treatment_level_dir, file_name, run_id)
            if not os.path.isfile(file_write_path):
                print(count, file)

                


                # if args.server_restart == 1:
                #     # purge cache
                #     sudoPassword = '<password>'
                #     command = 'sync && sudo purge'
                #     os.system('echo %s|sudo -S %s' % (sudoPassword, command))
                #     os.system("sudo purge")
                #     restart_postgres(config)
                   
                    

                conn = start_conn(config, timeout=args.timeout)
                with open(file_read_path, 'r') as f:
                    query = json.load(f)
                

                if args.dbname != "mathso" or query["failure"] == False:
                    try:
                        start_time = time.time()
                        json_result = run_query(query, conn, args.dbname, analyze=set_analyze_true)
                        end_time = time.time()
                        query["json_result"] = json_result
                        query["wall_clock_time (seconds)"] = end_time - start_time
                        query["index_level"] = index_level
                        query["memory_level"] = memory_level
                        query["page_cost_level"] = page_cost_level
                        query["treatment_config"] = treatment_conf
                        with open(file_write_path, "w", encoding='utf-8') as fp:
                            json.dump(query, fp, ensure_ascii=False, indent=4)
                        logging.info("{}/{} query executed".format(count, len(all_files)))
                    except QueryError:
                        logging.info("Failed executing {}".format(query["location"]))
                        with open(failure_file_name, "a") as fail_handle:
                            fail_handle.write(query["location"] + "\n") 
                        
                    conn.rollback()
                    conn.set_isolation_level(0)
                    cursor = conn.cursor()
                    cursor.execute("discard all")
                    cursor.close()
    else:
        # print("hi")
        query_id = args.query_id
        file = "postgres_query_{}.json".format(query_id) 
        file_read_path = "{}/{}".format(config["processed_queries_dir"], file)
        file_name = file.split(".")[0]
        run_id = 0
        file_write_path = "{}/{}_{}.json".format(treatment_level_dir, file_name, run_id)
        # print(file_write_path)
        while os.path.isfile(file_write_path):
            run_id = run_id + 1
            file_write_path = "{}/{}_{}.json".format(treatment_level_dir, file_name, run_id)
        # print(file_write_path)
        if not os.path.isfile(file_write_path):
            # stop_postgres(config)
            # start_postgres(config)
            # print(file)
            conn = start_conn(config, timeout=args.timeout)
            with open(file_read_path, 'r') as f:
                query = json.load(f)

            if query["failure"] == False:
                try:
                    start_time = time.time()
                    print("Running query {}".format(query["location"]))
                    json_result = run_query(query, conn)
                    end_time = time.time()
                    query["wall_clock_time (seconds)"] = end_time - start_time
                    query["json_result"] = json_result
                    query["index_level"] = index_level
                    query["memory_level"] = memory_level
                    query["page_cost_level"] = page_cost_level
                    query["treatment_config"] = treatment_conf
                    with open(file_write_path, "w", encoding='utf-8') as fp:
                        json.dump(query, fp, ensure_ascii=False, indent=4)
                    
                except QueryError:
                    logging.info("Failed executing {}".format(query["location"]))
                    with open(failure_file_name, "a") as fail_handle:
                        fail_handle.write(query["location"] + "\n") 
                    
                conn.rollback()
                conn.set_isolation_level(0)
                cursor = conn.cursor()
                cursor.execute("discard all")
                cursor.close()


def wait_select_inter(conn):
    while 1:
        try:
            state = conn.poll()
            if state == POLL_OK:
                break
            elif state == POLL_READ:
                select([conn.fileno()], [], [])
            elif state == POLL_WRITE:
                select([], [conn.fileno()], [])
            else:
                raise conn.OperationalError(
                    "bad state from poll: %s" % state)
        except KeyboardInterrupt:
            conn.cancel()
            # the loop will be broken by a server error
            continue

def run_query(query, conn, dbname , analyze=True):
    cur = conn.cursor()
    should_close = True
    query_text = query["query_text"]
    if query_text.startswith("create view"):
        create_view_statement, query_text, drop_view_statement, _ = query_text.split(";")
        cur.execute(create_view_statement)
        

    try:
        if analyze:
            statement = u"SET track_io_timing = ON; EXPLAIN (ANALYZE true, COSTS true, BUFFERS true, VERBOSE true, FORMAT json) \n {0}".format(query_text)
        else:
            statement = u"EXPLAIN (COSTS true, VERBOSE true, FORMAT json) \n {0}".format(query_text)

        if dbname == "mathso" and len(query["parameters"]) > 0:
            cur.execute(statement, query["parameters"])
        else:
            cur.execute(statement)
        
        if query_text.startswith("create view"):
            cur.execute(drop_view_statement)

        # output contained in first cell of result
        json_result = cur.fetchone()[0]
        if not [r for r in json_result if r["Plan"]["Plan Rows"] > 0]:
            print("No rows returned")
            # raise QueryError("No rows returned")
        return json_result
    except psycopg2.extensions.QueryCanceledError as e:
        # ctrl-c and a timeout will both raise this error
        # timeouts are "failures", ctrl-c are an exit condition
        should_close = False
        # if e.message.strip() == "canceling statement due to statement timeout":
        #     # ironically, in order to gracefully recover from this, we 
        #     # need to wait a few seconds before creating any new connections
        #     time.sleep(5)
        print(e)
        raise QueryError()
    except psycopg2.Error as e:
        print(e)
        raise QueryError()
    except KeyError as e:
        # this occurs if there is a parameter in the query we didn't substitute for
        print(e)
        raise QueryError()
    except TypeError as e:
        # this occurs if there is an unescaped percentage sign in the query
        print(e)
        raise QueryError()
    finally:
        if should_close:
            cur.close()

if __name__ == "__main__":
    main()
