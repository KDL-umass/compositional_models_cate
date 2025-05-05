# create a json file that contains table name and column names, number of rows and number of columns for all tables in the database
from data_utils import  start_conn, generate_basic_config
import json
import os
import argparse
from treatment_implementation import get_postgres_status, start_postgres
# arguments 
# --dbname: name of the database

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# generate arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type = str, required = True, help = "database name") # make this required
parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
args = parser.parse_args()


# load config containing database settings
config_name = "config.json"
config_path = "{}/queries/jsons".format(ROOT_DIR)
config_file_path = "{}/{}".format(config_path, config_name)
if not os.path.exists(config_path):
    os.makedirs(config_path)
config = generate_basic_config(config_file_path, args.dbname, args.data_folder_name)

# print config
print(config)

# connect to the database
status = get_postgres_status(config)
if status == False:
    start_postgres(config)
conn = start_conn(config)
cursor = conn.cursor()

# get all table names
cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';")
table_names = [u[0] for u in cursor.fetchall()]

# get all column names for each table
table_info = dict()
for table_name in table_names:
    table_info[table_name] = dict()
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{}'".format(table_name))
    columns = [u[0] for u in cursor.fetchall()]
    table_info[table_name]["columns"] = columns

# get number of rows in each table

for table_name in table_names:
    cursor.execute("SELECT count(*) from {}".format(table_name))
    num_rows = cursor.fetchall()[0][0]
    table_info[table_name]["num_rows"] = num_rows

# get number of columns in each table

for table_name in table_names:
    cursor.execute("SELECT COUNT(COLUMN_NAME) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_CATALOG = '{}' AND TABLE_SCHEMA = 'public' AND TABLE_NAME = '{}'".format(config["dbname"], table_name))
    num_cols = cursor.fetchall()[0][0]
    table_info[table_name]["num_cols"] = num_cols

for table_name in table_names:
    table_info[table_name]["primary_keys"] = []
    table_info[table_name]["foreign_keys"] = []

# create a json file that contains table name and column names, number of rows and number of columns for all tables in the database
# if data folder does not exist, create it
if not os.path.exists(config["data_folder_name"]):
    os.makedirs(config["data_folder_name"])
json_file_path = "{}/{}{}".format(config["data_folder_name"], config["dbname"], "_table_info.json")
with open(json_file_path, "w", encoding='utf-8') as fp:
    json.dump(table_info, fp, ensure_ascii=False, indent=4)
