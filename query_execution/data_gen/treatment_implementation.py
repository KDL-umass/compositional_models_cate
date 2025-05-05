# import psycopg2
# import psycopg2.extras
import json
import os
import argparse
import subprocess
import re
import shutil
import logging
from subprocess import Popen, PIPE, STDOUT
from io import StringIO 
from data_utils import *

SHARED_BUFFER_RE = re.compile("^shared_buffers\s*=\s*(?P<quantity>[0-9]+)(?P<measure>[A-z]+)\s*")
TEMP_BUFFER_RE = re.compile("^#?temp_buffers\s*=\s*(?P<quantity>[0-9]+)(?P<measure>[A-z]+)\s*")
WORK_MEM_RE = re.compile("^#?work_mem\s*=\s*(?P<quantity>[0-9]+)(?P<measure>[A-z]+)\s*")
SEQ_PAGE_COST_RE = re.compile("^#?seq_page_cost\s*=\s*(?P<quantity>[0-9\.]+)")
RANDOM_PAGE_COST_RE = re.compile("^#?random_page_cost\s*=\s*(?P<quantity>[0-9\.]+)")
PARALLEL_WORKERS_RE = re.compile("^#?max_parallel_workers\s*=\s*(?P<quantity>[0-9]+)")
PARALLEL_WORKERS_PER_GATHER_RE = re.compile("^#?max_parallel_workers_per_gather\s*=\s*(?P<quantity>[0-9]+)")
PARALLEL_MAINTENANCE_WORKERS_RE = re.compile("^#?max_parallel_maintenance_workers\s*=\s*(?P<quantity>[0-9]+)")
WORKER_PROCESSES_RE = re.compile("^#?max_worker_processes\s*=\s*(?P<quantity>[0-9]+)")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_postgres_status(config):
    proc = subprocess.Popen(["pg_ctl", "status", "-D", config["postgres_dir"]])
    stdout, stderr = proc.communicate()
    return proc.returncode == 0

def start_postgres(config):
    proc = Popen(["pg_ctl", "start", "-w", "-D", config["postgres_dir"]])
    output, error = proc.communicate()
    # log_subprocess_output(StringIO(output))
    return (proc.returncode, output, error)

def stop_postgres(config):
    proc = subprocess.Popen(["pg_ctl", "stop", "-m", "fast", "-D", config["postgres_dir"]])
    output, error = proc.communicate()
    # log_subprocess_output(StringIO(output))
    return (proc.returncode, output, error)

def restart_postgres(config):
    proc = subprocess.Popen(["pg_ctl", "restart", "-D", config["postgres_dir"]])
    output, error = proc.communicate()
    # log_subprocess_output(StringIO(output))
    return (proc.returncode, output, error)

def brew_start_postgres(config):
    proc = subprocess.Popen(["brew", "services", "start", "postgresql@14"])
    stdout, stderr = proc.communicate()
    return (proc.returncode, stdout, stderr)

def brew_restart_postgres(config):
    proc = subprocess.Popen(["brew", "services", "restart", "postgresql@14"])
    stdout, stderr = proc.communicate()
    return (proc.returncode, stdout, stderr)

def brew_stop_postgres(config):
    proc = subprocess.Popen(["brew", "services", "stop", "postgresql@14"])
    stdout, stderr = proc.communicate()
    return (proc.returncode, stdout, stderr)

def log_subprocess_output(pipe):
    for line in iter(pipe.readline, b''): # b'\n'-separated lines
        logging.info('got line from subprocess: %r', line)

def get_all_constraints(config, schemaname, tablename):
    conn = start_conn(config)
    query = """select con.conname
    from pg_catalog.pg_constraint con
        INNER JOIN pg_catalog.pg_class rel on rel.oid = con.conrelid
        INNER JOIN pg_catalog.pg_namespace nsp on nsp.oid = connamespace
        where nsp.nspname = '{0}'
        and rel.relname = '{1}';
    """.format(schemaname, tablename)
    # print(query)
    cursor = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cursor.execute(query)
    constraints = cursor.fetchall()
    return constraints

# returns the list of all the primary key and foreign key indices set for all the tables in the database. 
def get_index_level(config):
    conn = start_conn(config)
    query = """select t.relname as table_name, i.relname as index_name, a.attname
from
    pg_class t, pg_class i, pg_namespace n,
    pg_index ix, pg_attribute a
where
    t.oid = ix.indrelid and i.oid = ix.indexrelid
    and n.oid = i.relnamespace and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey) and t.relkind = 'r'
    and n.nspname = 'public' 
order by t.relname, i.relname
"""
    cursor = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cursor.execute(query)
    
    all_results = cursor.fetchall()
    print(all_results)
    
    dbname = config["dbname"]
    data_folder_name = config["data_folder_name"]
    data_json_file = os.path.join(data_folder_name, "{}_table_info.json".format(dbname))
    with open(data_json_file, "r") as handle:
        table_info = json.load(handle)
    conn.close()
    retval = 0
    
    if len(all_results) == 0:
        retval = 0
    else:
        
        pot_pk_idx = [r for r in all_results if r["table_name"] in table_info.keys() and r["attname"].lower() in table_info[r["table_name"]]["primary_keys"]]
        pot_f_idx = [r for r in all_results if r["table_name"] in table_info.keys() and r["attname"].lower() in [fk["column"] for fk in table_info[r["table_name"]]["foreign_keys"]]]
        pot_all_key_idx = []
        for table_name, table_values in table_info.items():
            all_columns = []
            if isinstance(table_values["columns"], dict):
                all_columns = list(table_values["columns"].keys())
            else:
                all_columns = table_values["columns"]
            for r in all_results:
                if r["table_name"] == table_name and r["attname"].lower() in all_columns:
                    pot_all_key_idx.append(r)
        
        if len(pot_pk_idx) > 0 and len(pot_f_idx) == 0:
            retval = 1
        else:
            if len(pot_pk_idx) == 0 and len(pot_f_idx) > 0:
                retval = 2
            else:
                if len(pot_pk_idx) > 0 and len(pot_f_idx) > 0 and (len(pot_all_key_idx) == len(pot_f_idx) + len(pot_pk_idx)):
                    retval = 3
                else:
                    if len(pot_all_key_idx) > len(pot_f_idx) + len(pot_pk_idx):
                        retval = 4
                    else:
                        retval = 0
    print("Index level: ", retval)     
    return retval

def set_index_level(config, index_level, index_type = "btree"):
    conn = start_conn(config)
    cursor = conn.cursor()
    
    query = """select schemaname, 
        indexname, 
        tablename, 
        format('drop index if exists %I.%I;', schemaname, indexname) as drop_statement
        from pg_indexes
    where schemaname not in ('pg_catalog', 'pg_toast');"""
    cursor.execute(query)
    all_results = cursor.fetchall()
    print(all_results)
    print("Index level: ", index_level)
    dbname = config["dbname"]
    data_folder_name = config["data_folder_name"]
    data_json_file = os.path.join(data_folder_name, "{}_table_info.json".format(dbname))
    with open(data_json_file, "r") as handle:
        table_info = json.load(handle)
    primary_keys = []
    foreign_keys = []
    all_keys = []
    if index_level > 1:
        for table_name, table_values in table_info.items():
            # print(table_name, table_values.keys())
            for pk in table_values["primary_keys"]:
                primary_keys.append((table_name, pk))
            for fk in table_values["foreign_keys"]:
                foreign_keys.append((table_name, fk["column"]))
            columns = table_values["columns"]
            # if columns is a dict 
            if isinstance(columns, dict):
                columns = list(columns.keys())
            for column_name in columns:
                all_keys.append((table_name, column_name))


    # drop all the indices
    if index_level == 0:
        for schemaname, indexname, tablename, drop_statement in all_results:
            print("Dropping index: ", drop_statement)
            all_constraints = get_all_constraints(config, schemaname, tablename)
            if(len(all_constraints)) > 0:
                for constraint_name in all_constraints[0]:
                    cursor.execute("ALTER TABLE {0} DROP CONSTRAINT IF EXISTS {1}".format(tablename, constraint_name))
            cursor.execute(drop_statement)
    else:
        if index_level == 1 or index_level == 3:
            # create primary keys 
            # cursor.execute("""select t.table_name, c.column_name from information_schema.tables t join information_schema.columns c 
            #                   on t.table_name = c.table_name where t.table_type = 'BASE TABLE' and c.table_schema = 'public' 
            #                   and c.column_name = 'id'""")
            # results = cursor.fetchall()
            # print(results)
            # print("Primary keys")
            # print(primary_keys)
            for table_name, column_name in primary_keys:
                if index_type == "btree":
                    # create index if not exists
                    create_statement = "create index if not exists {0}_{1}_{2}_ix on public.{0}({1})".format(table_name, column_name, index_type)
                    drop_statement = "drop index if exists {0}_{1}_{2}_ix".format(table_name, column_name, "hash")
                elif index_type == "hash":
                    create_statement = "create index if not exists {0}_{1}_{2}_ix on public.{0} using hash ({1})".format(table_name, column_name, index_type)
                    drop_statement = "drop index if exists {0}_{1}_{2}_ix".format(table_name, column_name, "btree")
                print(create_statement)
                try:
                    cursor.execute(drop_statement)
                    cursor.execute(create_statement)
                    conn.commit()
            
                except Exception as e:
                    print(e)
                    conn.commit()
                    cursor.close()
                    conn.close()

                    # restart postgres
                    restart_postgres(config)
                    conn = start_conn(config)
                    cursor = conn.cursor()

                    continue

    if index_level == 2 or index_level == 3:
        # create foreign keys 
        # cursor.execute("""select t.table_name, c.column_name from information_schema.tables t join information_schema.columns c 
        #                   on t.table_name = c.table_name where t.table_type = 'BASE TABLE' and c.table_schema = 'public' 
        #                   and column_name like '%_id'
        #                 """)
        # print("Foreign keys")
        # print(foreign_keys)
        for table_name, column_name in foreign_keys:
            # print(table_name, column_name)
            if index_type == "btree":
                # create index if not exists
                create_statement = "create index if not exists {0}_{1}_{2}_ix on public.{0}({1})".format(table_name, column_name, index_type)
            elif index_type == "hash":
                create_statement = "create index if not exists {0}_{1}_{2}_ix on public.{0} using hash ({1})".format(table_name, column_name, index_type)
            print(create_statement)
            try:
                cursor.execute(create_statement)
                conn.commit()
            except Exception as e:
                print(e)
                conn.commit()
                cursor.close()
                conn.close()

                # restart postgres
                restart_postgres(config)
                conn = start_conn(config)
                cursor = conn.cursor()
                continue

    if index_level == 4:
        # print("All keys")
        for table_name, column_name in all_keys:
            # print(table_name, column_name)
            if index_type == "btree":
                # create index if not exists
                create_statement = "create index if not exists {0}_{1}_{2}_ix on public.{0}({1})".format(table_name, column_name, index_type)
            elif index_type == "hash":
                create_statement = "create index if not exists {0}_{1}_{2}_ix on public.{0} using hash ({1})".format(table_name, column_name, index_type)
            print(create_statement)
            try:
                cursor.execute(create_statement)
                conn.commit()
            except Exception as e:
                print(e)
                conn.commit()
                cursor.close()
                conn.close()

                # restart postgres
                restart_postgres(config)
                conn = start_conn(config)
                cursor = conn.cursor()
                continue


           
    
    conn.commit()
    cursor.close()
    conn.close()

def get_shared_buffers(conf_lines):
    for line in conf_lines:
        match = re.match(SHARED_BUFFER_RE, line.strip())
        if match:
            return get_byte_count(match.groupdict()["quantity"], 
            match.groupdict()["measure"])

def get_work_mem(conf_lines):
    for line in conf_lines:
        match = re.match(WORK_MEM_RE, line.strip())
        if match:
            return get_byte_count(match.groupdict()["quantity"], 
            match.groupdict()["measure"])

def get_temp_buffers(conf_lines):
    for line in conf_lines:
        match = re.match(TEMP_BUFFER_RE, line.strip())
        if match:
            return get_byte_count(match.groupdict()["quantity"], 
            match.groupdict()["measure"])

def get_seq_page_cost(conf_lines):
    for line in conf_lines:
        match = re.match(SEQ_PAGE_COST_RE, line.strip())
        if match:
            return float(match.groupdict()["quantity"])
            
def get_random_page_cost(conf_lines):
    for line in conf_lines:
        match = re.match(RANDOM_PAGE_COST_RE, line.strip())
        if match:
            return float(match.groupdict()["quantity"])

def get_byte_count(quantity, measure):
    quantity = int(quantity)
    multipliers = {
                    "kb" : 1000,
                    "mb" : 1000 ** 2,
                    "gb": 1000 ** 3
    }
    return quantity * multipliers[measure.lower()]

def get_size_description(byte_quantity):
    if byte_quantity > (1000 ** 2): # 1 M megabytes:
        return "{0}MB".format(int(byte_quantity/(1000 ** 2)))
    elif byte_quantity > 1000: # kilobytes
        return "{0}kB".format(int(byte_quantity/1000))
    else:
        return "{0}b".format(byte_quantity)
    
def get_parallelization_level(conf_lines):
    for line in conf_lines:
        match = re.match(PARALLEL_WORKERS_RE, line.strip())
        if match:
            return int(match.groupdict()["quantity"])

def get_enable_indexscan(conf_lines):
    for line in conf_lines:
        # if line is commented out, then index scan is enabled
        match = re.match("#enable_indexscan", line.strip())
        if match:
            return "off"
        match = re.match("enable_indexscan = on", line.strip())
        if match:
            return "on"
    return "not found"

def get_index_type(config):
    conn = start_conn(config)
    cursor = conn.cursor()
    cursor.execute("""select p.indexdef from pg_indexes p where schemaname = 'public' """)
    all_results = cursor.fetchall()
    conn.close()
    if len(all_results) == 0:
        return "no index"
    else:
        
        index_def = all_results[0][0]
        # print(index_def)
        # lower case the index def
        index_def = index_def.lower()
        if "using hash" in index_def:
            return "hash"
        else:
            if "using btree" in index_def:
                return "btree"
            else:
                return "no index"


         
def get_treatment_config(config):
    print("Getting treatment config")
    psql_config_file = os.path.join(config["postgres_dir"], "postgresql.conf")
    with open(psql_config_file, "r") as handle:
        psql_conf_lines = handle.readlines()
    
    shared_buffers = get_shared_buffers(psql_conf_lines)
    temp_buffers = get_temp_buffers(psql_conf_lines)
    work_mem = get_work_mem(psql_conf_lines)
    random_page_cost = get_random_page_cost(psql_conf_lines)
    seq_page_cost = get_seq_page_cost(psql_conf_lines)
    print("shared buffers: ", shared_buffers)
    print("temp buffers: ", temp_buffers)
    print("work mem: ", work_mem)
    print("random page cost: ", random_page_cost)
    print("seq page cost: ", seq_page_cost)
    index_level = get_index_level(config)
    print("index level: ", index_level)
    memory_level = get_memory_level(shared_buffers, temp_buffers, work_mem)
    print("memory level: ", memory_level)
    page_cost = get_page_cost(random_page_cost, seq_page_cost)
    print("page cost: ", page_cost)
    shared_buffers = get_size_description(shared_buffers)
    print("shared buffers: ", shared_buffers)
    temp_buffers = get_size_description(temp_buffers)
    print("temp buffers: ", temp_buffers)
    work_mem = get_size_description(work_mem)
    print("work mem: ", work_mem)
    parallelization = get_parallelization_level(psql_conf_lines)
    print("parallelization: ", parallelization)
    enable_indexscan = get_enable_indexscan(psql_conf_lines)
    print("enable index scan: ", enable_indexscan)
    index_type = "hash"
    # Taking too long to run; skipping this for now
    # index_type = get_index_type(config)

    return {
        "index_level": index_level,
        "memory_level": memory_level,
        "page_cost": page_cost,
        "shared_buffers": shared_buffers,
        "temp_buffers": temp_buffers,
        "work_mem": work_mem,
        "parallelization": parallelization,
        "enable_indexscan": enable_indexscan,
        "index_type": index_type
    }

def set_treatment_config(config, index_level, memory_level, page_cost, disable_parallelization = False, set_index_level_flag = True, set_memory_level_flag = True, set_page_cost_level_flag = True, enable_indexscan = False, index_type = "btree"):
    """
    Keyword arguments:
    index_level --
        0: no indexing
        1: indexing on PK,
        2: indexing on PK and FK
    memory_level --
        0: work_memory and buffer memory at minimum,
        1: moderate memory,
        2: high work/buffer memory
    random page cost --
        0: relative high random page cost,
        1: random/sequence cost balanced,
        2: relatively high sequence page cost 
    """
    print("Setting treatment config")
    if set_memory_level_flag or set_page_cost_level_flag or disable_parallelization:

        psql_conf_file = os.path.join(config["postgres_dir"], "postgresql.conf")
        with open(psql_conf_file, "r") as handle:
            postgres_config_lines = handle.readlines()

        if memory_level == 0:
            temp_buffers = 800 * 1000 #800 KB min
            work_mem = 64 * 1000 #64 KB min 
        elif memory_level == 1:
            temp_buffers = 8 * (1000**2) # 8MB
            work_mem = 1000**2 # 1MB
        elif memory_level == 2:
            temp_buffers = 100 * (1000** 2) # 100MB
            work_mem = 50 * (1000 ** 2) # 50MB

        if page_cost == 0:
            random_page_cost = 5.0
            seq_page_cost = 1.0
        elif page_cost == 1:
            random_page_cost = 2.0
            seq_page_cost = 2.0
        elif page_cost == 2:
            random_page_cost = 1.0
            seq_page_cost = 5.0

        if disable_parallelization:
            max_parallel_workers_per_gather = 0
            max_parallel_workers = 0
            max_worker_processes = 0
            max_parallel_maintenance_workers = 0
            for i, line in enumerate(postgres_config_lines):
                if re.match(PARALLEL_WORKERS_PER_GATHER_RE, line.strip()):
                    postgres_config_lines[i] = "max_parallel_workers_per_gather = {0}\n".format(max_parallel_workers_per_gather)
                if re.match(PARALLEL_WORKERS_RE, line.strip()):
                    postgres_config_lines[i] = "max_parallel_workers = {0}\n".format(max_parallel_workers)
                if re.match(WORKER_PROCESSES_RE, line.strip()):
                    postgres_config_lines[i] = "max_worker_processes = {0}\n".format(max_worker_processes)
                if re.match(PARALLEL_MAINTENANCE_WORKERS_RE, line.strip()):
                    postgres_config_lines[i] = "max_parallel_maintenance_workers = {0}\n".format(max_parallel_maintenance_workers)

        if enable_indexscan:
            # uncomment enable_indexscan
            for i, line in enumerate(postgres_config_lines):
                if re.match("#?enable_indexscan", line.strip()):
                    postgres_config_lines[i] = "enable_indexscan = on\n"


        # else:
        #     max_parallel_workers_per_gather = 2
        #     max_parallel_workers = 2
        #     max_worker_processes = 4
        #     max_parallel_maintenance_workers = 2
        #     for i, line in enumerate(postgres_config_lines):
        #         if re.match("max_parallel_workers_per_gather", line.strip()):
        #             postgres_config_lines[i] = "max_parallel_workers_per_gather = {0}\n".format(max_parallel_workers_per_gather)
        #         if re.match("max_parallel_workers", line.strip()):
        #             postgres_config_lines[i] = "max_parallel_workers = {0}\n".format(max_parallel_workers)
        #         if re.match("max_worker_processes", line.strip()):
        #             postgres_config_lines[i] = "max_worker_processes = {0}\n".format(max_worker_processes)
        #         if re.match("max_parallel_maintenance_workers", line.strip()):
        #             postgres_config_lines[i] = "max_parallel_maintenance_workers = {0}\n".format(max_parallel_maintenance_workers)

        for i, line in enumerate(postgres_config_lines):
            if re.match(TEMP_BUFFER_RE, line.strip()):
                postgres_config_lines[i] = "temp_buffers = {0}\n".format(get_size_description(temp_buffers))
            
            if re.match(WORK_MEM_RE, line.strip()):
                postgres_config_lines[i] = "work_mem = {0}\n".format(get_size_description(work_mem))

            if re.match(RANDOM_PAGE_COST_RE, line.strip()):
                postgres_config_lines[i] = "random_page_cost = {0}\n".format(random_page_cost)

            if re.match(SEQ_PAGE_COST_RE, line.strip()):
                postgres_config_lines[i] = "seq_page_cost = {0}\n".format(seq_page_cost)

        # write old config file as backup 
        shutil.copyfile(psql_conf_file, psql_conf_file + ".bak")
        with open(psql_conf_file, "w+") as handle:
            handle.writelines(postgres_config_lines)
    

    if set_index_level_flag:
        print("Setting index level")
        set_index_level(config, index_level, index_type = index_type)
    print("Setting treatment config done")

def get_memory_level(shared_buffers, temp_buffers, work_mem):
    if temp_buffers == 800 * 1000 and work_mem == 64 * 1000:
        memory_level = 0
    elif temp_buffers == 8 * (1000 **2) and work_mem == (1000 **2):
        memory_level = 1
    elif temp_buffers == 100 * 1000 ** 2 and work_mem == 50 * 1000 ** 2:
        memory_level = 2
    return memory_level

def get_page_cost(random_page_cost, seq_page_cost):
    if random_page_cost == 5.0 and seq_page_cost == 1.0:
        return 0
    elif random_page_cost == 2.0 and seq_page_cost == 2.0:
        return 1
    elif random_page_cost == 1.0 and seq_page_cost == 5.0:
        return 2
    


def restart_postgres(config):
    # rc, stdout, stderr = brew_stop_postgres(config)
    # rc, stdout, stderr = brew_restart_postgres(config)
    rc, stdout, stderr = stop_postgres(config)
    rc, stdout, stderr = start_postgres(config)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dbname', type = str, default = "stackoverflow2014", help = "database name")
    parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
    parser.add_argument('--index_level', type = int, default = 3, help = "sets index level")
    parser.add_argument('--memory_level', type = int, default = 0, help = "sets memory level")
    parser.add_argument('--page_cost_level', type = int, default = 0, help = "sets the level of page costs")
    parser.add_argument('--disable_parallelization', type = int, default = 0, help = "should parallelism be disabled")
    parser.add_argument('--enable_indexscan', type = int, default = 1, help = "should index scan be enabled")
    parser.add_argument('--index_type', type = str, default = "hash", help = "index type to be used")

    args = parser.parse_args()
    dbname = args.dbname
    data_folder_name = args.data_folder_name
    index_level = args.index_level
    memory_level = args.memory_level
    page_cost = args.page_cost_level
    disable_parallelization = args.disable_parallelization
    
    # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(ROOT_DIR)
    config_file_path = "{}/{}".format(config_path, config_name)
    config = generate_basic_config(config_file_path, dbname, data_folder_name)
    treatment_conf = get_treatment_config(config)
    
    set_treatment_config(config, index_level, memory_level, page_cost, disable_parallelization = disable_parallelization, set_index_level_flag = True, set_memory_level_flag = True, set_page_cost_level_flag = True, enable_indexscan = args.enable_indexscan, index_type = args.index_type)
    treatment_conf = get_treatment_config(config)
    

    
    
    
if __name__ == '__main__':
    main()