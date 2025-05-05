import requests
import requests_cache
from bs4 import BeautifulSoup
import re
import json
import os
import logging
import argparse
import pandas as pd
from data_utils import  generate_basic_config, return_query_id_from_link
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dbname', type = str, required = True, help = "database name")
parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
parser.add_argument('--num_pages', type = int,  default = 1, help = "number of pages to pull queries from")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

args = parser.parse_args()
dbname = args.dbname
data_folder_name = args.data_folder_name
num_pages = args.num_pages

config_name = "config.json"
config_path = "{}/queries/jsons".format(ROOT_DIR)
config_file_path = "{}/{}".format(config_path, config_name)

config = generate_basic_config(config_file_path, args.dbname, args.data_folder_name)

log_dir = config["logs_dir"]
user_queries_dir = config["user_queries_dir"]
if not os.path.exists(user_queries_dir):
    os.makedirs(user_queries_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


logname = "{}/{}".format(log_dir, "pull_queries.out")
logging.basicConfig(filename=logname,
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

requests_cache.install_cache("so_cache", allowable_codes = (200, ))
query_link_pattern = re.compile("/stackoverflow/query/[0-9]+")
user_pattern = re.compile("/users/[0-9]+")



def main():
    
    for page_num in range(0, num_pages):
        logging.info("Page number: {}".format(page_num))
        all_urls = get_query_links(page_num)
        print(len(all_urls))
        
        count = 0
        timeout = 900
        for link in all_urls:
            logging.info("{0}".format(link))
            query_id = return_query_id_from_link(link)
            file_name = "{}/query_{}.json".format(user_queries_dir, query_id)
            response_code = 406
            if os.path.isfile(file_name):
                with open(file_name, 'r') as f:
                    already_query = json.load(f)
                response_code = already_query["response_code"]
                while response_code != 200:
                    query_info = process_query(link)
                    response_code = query_info["response_code"]
                    if response_code == 200:
                        with open(file_name, "w", encoding='utf-8') as fp:
                            json.dump(query_info, fp, ensure_ascii=False, indent=4)
                            logging.info("DONE {}, {}, {}".format(count, query_id, link))
                    else:
                        print("Sleeping for {} minutes".format(int(timeout/60)))
                        time.sleep(timeout)
            else:
                while response_code != 200:
                    query_info = process_query(link)
                    response_code = query_info["response_code"]
                    with open(file_name, "w", encoding='utf-8') as fp:
                        json.dump(query_info, fp, ensure_ascii=False, indent=4)
                        logging.info("DONE {}, {}, {}".format(count, query_id, link))
                    if response_code != 200:
                        print("Sleeping for {} minutes".format(int(timeout/60)))
                        time.sleep(timeout)
            count += 1

def process_query(query_link):
    resp = requests.get(query_link)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, features="lxml")

        query_form = soup.find(id="query")
        favorite_count = int(query_form.find("div", {"class" : "favoritecount"}).get_text())
        description = query_form.find("p", {"class" : "description"}).get_text()
        query_text = query_form.find(id="queryBodyText").get_text()

        meta_table = soup.find("table", {"class" : "fw"})
        sig_col = meta_table.find("div", {"class" : "user-info owner"})
        create_time = sig_col.find("span", {"class" : "relativetime"})["title"]

        edited = len(meta_table.find_all("div", {"class" : "user-info"})) > 1

        user_link = sig_col.find("a")
        if not user_link:
            user_href = None
            user_name = "anonymous"
        else:
            user_href = user_link["href"]
            user_name = sig_col.find_all("a", href=user_pattern)[-1].get_text()

        param_div = soup.find(id="query-params")
        param_names = [p["name"] for p in param_div.find_all("input")]

        query_info = {
                        "location" : query_link,
                        "favorite_count" : favorite_count,
                        "description" : description,
                        "query_text" : query_text,
                        "create_time" : create_time,
                        "user_href" : user_href,
                        "user_name" : user_name,
                        "edited" : edited,
                        "response_code": resp.status_code
                    }
        return query_info
    else:
        return {"location": query_link, "response_code": resp.status_code}

def get_query_links(pgnum):
    """ Scrapes information about queries on a particular page of the SO site """
    resp = requests.get("http://data.stackexchange.com/stackoverflow/queries?order_by=popular&pagesize=100&page={0}".format(pgnum))
    soup = BeautifulSoup(resp.text)
    relevant_links = soup.findAll("a", href=query_link_pattern)
    absolute_links = ["http://data.stackexchange.com{0}".format(a["href"]) for a in relevant_links]
    return absolute_links



if __name__ == "__main__":
    main()


