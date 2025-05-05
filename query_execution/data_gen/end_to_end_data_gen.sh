
# first generate a metadata json file for the SO data that contains table names, table sizes, and table column names, 
# python generate_schema_db.py --dbname mathso --data_folder_name mathso_test
# note we will need to add the indices later manually in the generated metadata json file
# 
# # pull user queries in MSSQL from https://data.stackexchange.com/stackoverflow/queries?order_by=popular
# python pull_user_queries.py --dbname mathso --data_folder_name mathso_test

# rewrite pulled user queries into postgres
python rewrite_queries.py --dbname mathso --data_folder_name mathso_test

# parameterize rewritten user queries
python parameterize_queries.py --dbname mathso --data_folder_name mathso_test

# generate features from the query that are useful for query execution
python query_featurizer.py --dbname mathso --data_folder_name mathso_test

# execute the parameterized queries on the database for different index levels, memory levels, and page costs. Refer treatment_implementation.py to see how the different levels are defined.
for ix in 0;
    do for ml in 0 2;
        do for pg in 0;
            do 
                echo "$ix $ml $pg"
                python query_execution.py --dbname mathso --index_level $ix --memory_level $ml --page_cost_level $pg --data_folder_name mathso_test --timeout 300000 --rerun 1 --disable_parallelization 1
            done
        done
    done

# # # calculate individual component-level outcomes and generate csvs for component-level modeling
# python post_process_queries.py --dbname mathso --data_folder_name mathso_test  --rerun 1
# ## generate csvs for unit-level modeling. Results are saved in the queries/data_folder_name/data/all_csvs folder
# python collect_data_modeling_high_level.py --dbname mathso --data_folder_name mathso_test
