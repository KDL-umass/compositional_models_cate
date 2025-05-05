from os import listdir
import os
from os.path import isfile, join
import re
import json
import argparse
import logging
from data_utils import  generate_basic_config, set_logging


REGEX_REWRITE_RULES = [
                        # remove semicolons from end of statements
                        # this strengthens the top x rewrite rule
                        (re.compile(r";\s*$"), ""),

                        # remove lines containing only comments
                        # this strengthens the top x rewrite rule
                        (re.compile(r"\s*\-\-.+"), ""),

                    
                        # sql server wraps identifier names with brackets 
                        # postgres wraps identifiers with quotes
                        (re.compile(r"\[(?P<identifier>.+)\]", flags=re.IGNORECASE), 
                                    '"\g<identifier>"'),
                        
                        # convert "top x" statements to "limit x" statements
                        # this only works for the top-level query. Sub-queries would be a lot of work
                        (re.compile(r"^\s*select\s+top\s+(?P<topnum>[0-9]+)(?P<whitespace>\s+)(?P<querybody>.+)", flags=re.IGNORECASE | re.DOTALL), 
                                    'select \n \g<querybody> limit \g<topnum>'),

                        # change % to pct, since % is a special psycopg2 character (when contained in quotes as for a field name)
                        (re.compile('"(?P<before>.*?)%(?P<after>.*?)"'), '"\g<before>pct\g<after>"'),

                        # remove "declare" statements
                        (re.compile("declare\s*.+", flags=re.IGNORECASE), ""),

                        # re-format parameter names 
                        (re.compile("@(?P<paramname>[A-z]+)"), "%(\g<paramname>)s"),
                        (re.compile("##(?P<paramname>[A-z]+)##"), "%(\g<paramname>)s"),

                      ]


class Replacement(object):

    def __init__(self, replacement):
        self.replacement = replacement
        self.occurrences = []

    def __call__(self, match):
        matched = match.group(0)
        replaced = match.expand(self.replacement)
        self.occurrences.append((matched, replaced))
        return replaced
    
def rewrite_query(query, verbose=True):
    """ Process the query text """

    for regex, replacement in REGEX_REWRITE_RULES:
        repl = Replacement(replacement)
        query = regex.sub(repl, query)

        if verbose:
            if repl.occurrences:
                print("---------------------------")

            for match, replaced in repl.occurrences:
                print(u"{0} => {1}".format(match, replaced))
    return query

def return_query_id(file_name):
    match = re.search(r'query_(\w+)', file_name) 
    if match:                                    # Check if there is a match first
        return match.group(1)
    else:
        return -1


def main():
    # for now we are storing the same user queries used by previous NeurIPS and ICML papers. 
    # This can change if needed 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', type = str, default = "mathso", help = "database name")
    parser.add_argument('--data_folder_name', type = str, required = True, help = "name of the data folder")
    args = parser.parse_args()
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

     # load config containing database settings
    config_name = "config.json"
    config_path = "{}/queries/jsons".format(ROOT_DIR)
    config_file_path = "{}/{}".format(config_path, config_name)

    
    config = generate_basic_config(config_file_path, args.dbname, args.data_folder_name)
   

    log_dir = config["logs_dir"]
    user_queries_dir = config["user_queries_dir"]
    rewritten_queries_dir = config["rewritten_queries_dir"]
    if not os.path.exists(rewritten_queries_dir):
        os.makedirs(rewritten_queries_dir)

    logname = "{}/logs.out".format(log_dir)
    set_logging(logname)

    all_files = [f for f in listdir(user_queries_dir) if isfile(join(user_queries_dir, f))]
    
        
    print("Total number of files in {} dir {}".format(config["user_queries_dir"], len(all_files)))
    cnt = 0
    for file in all_files:
        count = return_query_id(file)
        read_file_name = "{}/{}".format(user_queries_dir, file)
        write_file_name = "{}/postgres_query_{}.json".format(rewritten_queries_dir, count)
        if not os.path.isfile(write_file_name):
            with open(read_file_name, 'r') as f:
                query = json.load(f)
            if query["response_code"] == 200:
                new_query = query
                try:
                    new_query["query_text"] = rewrite_query(query["query_text"], verbose = False)
                except Exception as e:
                    print(count, e)

                new_query["format"] = "postgresql"
                
                with open(write_file_name, "w", encoding='utf-8') as fp:
                    json.dump(new_query, fp, ensure_ascii=False, indent=4)
                logging.info("Write file {} count {} to {}".format(file, cnt, write_file_name))
                cnt += 1 

if __name__ == "__main__":
    main()





