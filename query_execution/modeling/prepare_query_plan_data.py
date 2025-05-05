import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import os
import json
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns

# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# define root dir as one level up
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# have one path up
ROOT_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR_ROOT = "{}/data_gen/queries/data".format(ROOT_DIR)
print(DATA_DIR_ROOT)
import sys
sys.path.append("{}/data_gen".format(ROOT_DIR))

def preprocess_index_scan_data(df, log_transform=True):
    if df is None:
        return None, None, None, None, None, None, None, None, None
    ir = df["input_rows"].values.reshape(-1,1)
    selectivity = df["output_rows"].values / df["input_rows"].values
    selectivity = selectivity.reshape(-1,1)
    pr = df["plan_rows"].values.reshape(-1,1)
    # lq = df["limit_query"].values.reshape(-1,1)
    # fc = df["filter_query"].values.reshape(-1,1)
    or_ = df["output_rows"].values.reshape(-1,1)
    df["shared_read_blocks"] = df["shared_read_blocks"].apply(lambda x: 0 if x < 0 else x)
    srb = df["shared_read_blocks"].values.reshape(-1,1)
    et = df["execution_time"].values.reshape(-1,1)

    if log_transform:
        ir = np.log10(ir+1)
        pr = np.log10(pr+1)
        or_ = np.log10(or_+1)
        
        srb = np.log10(srb+1)
        et = np.log10(et+1)
    # convert to torch tensor
    
    
    
    return ir, pr, or_, srb, et, selectivity, df

def preprocess_seq_scan_data(df, log_transform=True):
    if df is None:
        return None, None, None, None, None, None, None, None, None
    ir = df["input_rows"].values.reshape(-1,1)
    # srb > 0
    
    selectivity = df["output_rows"].values / df["input_rows"].values
    selectivity = selectivity.reshape(-1,1)

    
    pr = df["plan_rows"].values.reshape(-1,1)
    
    # lq = df["limit_query"].values.reshape(-1,1)
    # fc = df["filter_query"].values.reshape(-1,1)
    
    or_ = df["output_rows"].values.reshape(-1,1)
    df["shared_read_blocks"] = df["shared_read_blocks"].apply(lambda x: 0 if x < 0 else x)
    
    srb = df["shared_read_blocks"].values.reshape(-1,1)
    
    et = df["execution_time"].values.reshape(-1,1)

    if log_transform:
        ir = np.log10(ir+1)
        pr = np.log10(pr+1)
        or_ = np.log10(or_+1)
        
        
        srb = np.log10(srb+1)
        et = np.log10(et+1)
    
    return ir, pr, or_, srb, et, selectivity, df

def preprocess_hash_data(df, log_transform=True):
    if df is None:
        return None, None, None, None, None
    ir = df["input_rows"].values.reshape(-1,1)
    
    # convert to torch tensor

    pr = df["plan_rows"].values.reshape(-1,1)
    
    # hash_buckets = df["Original Hash Buckets"].values.reshape(-1,1)

    or_ = df["output_rows"].values.reshape(-1,1)
    
    et = df["execution_time"].values.reshape(-1,1)
    
    if log_transform:
        ir = np.log10(ir+1)
        pr = np.log10(pr+1)
        or_ = np.log10(or_+1)
        et = np.log10(et+1)
    
    return ir, pr, or_, et, df
def preprocess_hash_join_data(df, log_transform=True):
    if df is None:
        return None, None, None, None, None, None, None
    lr = df["left_output_rows"].values.reshape(-1,1)
    rr = df["right_output_rows"].values.reshape(-1,1)
    pr = df["plan_rows"].values.reshape(-1,1)
    or_ = df["output_rows"].values.reshape(-1,1)
    et = df["execution_time"].values.reshape(-1,1)
    
    join_type = df["join_type"].values.reshape(-1,1)
    # # convert to categorical
    # join_type_unique = np.unique(join_type)
    # join_type_dict = {join_type_unique[i]: i for i in range(len(join_type_unique))}
    # join_type = np.array([join_type_dict[jt] for jt in join_type]).reshape(-1,1)

    if log_transform:
        lr = np.log10(lr+1)
        rr = np.log10(rr+1)
        pr = np.log10(pr+1)
        or_ = np.log10(or_+1)
        et = np.log10(et+1)

    return lr, rr, pr, join_type, or_, et, df

def preprocess_sort_data(df, log_transform=True):
    if df is None:
        return None, None, None, None, None, None
    ir = df["input_rows"].values.reshape(-1,1)
    pr = df["plan_rows"].values.reshape(-1,1)
    df["sort_method"] = df["sort_method"].fillna("top-N heapsort")
    sort_method_pd = df["sort_method"].values.reshape(-1,1)
    # convert to categorical
    sort_method_unique = np.unique(sort_method_pd)
    sort_method_dict = {sort_method_unique[i]: i for i in range(len(sort_method_unique))}
    sort_method = []
    for sm in sort_method_pd:
        try:
            sort_method.append(sort_method_dict[sm])
        except:
            sort_method.append(sort_method_dict[sm[0]])
    sort_method = np.array(sort_method).reshape(-1,1)
    or_ = df["output_rows"].values.reshape(-1,1)
    et = df["execution_time"].values.reshape(-1,1)
    

    if log_transform:
        ir = np.log10(ir+1)
        pr = np.log10(pr+1)
        or_ = np.log10(or_+1)
        et = np.log10(et+1)

    # print(ir.shape, pr.shape, sort_method.shape, or_.shape, et.shape)

    return ir, pr, sort_method, or_, et, df

def preprocess_aggregate_data(df, log_transform=True):
    if df is None:
        return None, None, None, None, None, None
    ir = df["input_rows"].values.reshape(-1,1)
    pr = df["plan_rows"].values.reshape(-1,1)
    df["strategy"] = df["strategy"].fillna("Plain")
    aggregate_strategy_pd = df["strategy"].values.reshape(-1,1)
    # convert to categorical
    aggregate_strategy_unique = np.unique(aggregate_strategy_pd)
    aggregate_strategy_dict = {aggregate_strategy_unique[i]: i for i in range(len(aggregate_strategy_unique))}
    aggregate_strategy = []
    for sm in aggregate_strategy_pd:
        try:
            aggregate_strategy.append(aggregate_strategy_dict[sm])
        except:
            aggregate_strategy.append(aggregate_strategy_dict[sm[0]])
    aggregate_strategy = np.array(aggregate_strategy).reshape(-1,1)
    or_ = df["output_rows"].values.reshape(-1,1)
    et = df["execution_time"].values.reshape(-1,1)
    

    if log_transform:
        ir = np.log10(ir+1)
        pr = np.log10(pr+1)
        or_ = np.log10(or_+1)
        et = np.log10(et+1)

    # print(ir.shape, pr.shape, sort_method.shape, or_.shape, et.shape)

    return ir, pr, aggregate_strategy, or_, et, df

def get_hash_df(treatment_id=None, filters=True, data_folder_name="mathso", data_dir = "all_csvs", function_type="non_linear"):
    
    df = pd.read_csv('{}/{}/{}/all_hash_data.csv'.format(DATA_DIR_ROOT, data_folder_name, data_dir))

    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]
    # print(df[df['query_id'] == 113603][["query_id", "treatment_id", "Actual Loops"]])
    if filters:
        df = df[df["Actual Loops"] == 1]
    df.rename(columns = {'Self Shared Read Blocks':'shared_read_blocks', 'Plan Rows': 'plan_rows', 'Original Hash Buckets': 'hash_buckets'}, inplace = True)
    if "execution_time" not in df.columns:
        df.rename(columns = {'Self Time':'execution_time'}, inplace = True)
    
    if treatment_id is not None:
        if isinstance(treatment_id, list):
            df = df[df["treatment_id"].isin(treatment_id)]
        else:
            df = df[df["treatment_id"] == treatment_id]

    
        
    relevant_features = ['query_id', 'treatment_id','input_rows', 'plan_rows',  'output_rows', 'execution_time']
    
    relevant_features += ['hash_buckets',  'run_id', 'dbname', 'data_folder_name']
    df_hash = df[relevant_features]
    # print(df_hash[df_hash['query_id'] == 113603])
    return df_hash

def get_hash_train_test_split(train_query_ids = None, test_query_ids = None, treatment_id=None, log_transform=True, function_type="non_linear", plan_rows=False, noise_variance=0.0, data_folder_name="mathso", data_dir = "all_csvs"):
    df_hash = get_hash_df(treatment_id=treatment_id, function_type=function_type, data_folder_name=data_folder_name, data_dir=data_dir)
    if df_hash is None:
        return None, None, None, None, None, None, None

    ir, pr, or_, et, df_hash = preprocess_hash_data(df_hash, log_transform=log_transform)
    # reset index
    df_hash = df_hash.reset_index(drop=True)
    if train_query_ids is not None and test_query_ids is not None:
        tr_idx = df_hash[df_hash["query_id"].isin(train_query_ids)].index
        te_idx = df_hash[df_hash["query_id"].isin(test_query_ids)].index
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df_hash)), test_size=0.2, random_state=42)
    ir_tr, pr_tr, or_tr, et_tr = ir[tr_idx], pr[tr_idx], or_[tr_idx], et[tr_idx]
    ir_te, pr_te, or_te, et_te = ir[te_idx], pr[te_idx], or_[te_idx], et[te_idx]
    train_ids = df_hash.iloc[tr_idx]["query_id"].values
    test_ids = df_hash.iloc[te_idx]["query_id"].values
    train_treatment_ids = df_hash.iloc[tr_idx]["treatment_id"].values
    test_treatment_ids = df_hash.iloc[te_idx]["treatment_id"].values

    pr_tr_std = or_tr + noise_variance * (pr_tr - or_tr)
    pr_te_std = or_te + noise_variance * (pr_te - or_te)
    if plan_rows:
        X_train = np.concatenate((ir_tr, pr_tr_std), axis=1)
        X_test = np.concatenate((ir_te, pr_te_std), axis=1)
    else:
        X_train = np.concatenate((ir_tr, or_tr), axis=1)
        X_test = np.concatenate((ir_te, or_te), axis=1)
    y_train = et_tr
    y_test = et_te

    # scale data
    if len(X_train) > 0 and len(X_test) > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # y_train = scaler.fit_transform(y_train)
        # y_test = scaler.transform(y_test)
    

        # convert to torch tensor
        X_train = torch.tensor(X_train, dtype=torch.float)
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
    all_indices = np.concatenate([tr_idx, te_idx])
    df_hash = df_hash.iloc[all_indices]
    return X_train, X_test, y_train, y_test, df_hash, train_ids, test_ids, train_treatment_ids, test_treatment_ids

def get_hash_obs_df(query_treatment_ids = None, filters=True, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs"):
    # get all treatment ids
    df_hash = get_hash_df(filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    # print(df_hash[df_hash['query_id'] == 113603])
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)

    # filter treatment ids and query ids
    df_hash = df_hash[df_hash["query_id"].isin(query_ids)]
    df_hash = df_hash[df_hash["treatment_id"].isin(treatment_ids)]

    #  # get those query ids that have atleast two treatments
    # query_ids = df_hash.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    # # filter treatment ids and query ids
    # df_hash = df_hash[df_hash["query_id"].isin(query_ids)]

    
    # reset index
    df_hash = df_hash.reset_index(drop=True)
    print(df_hash.shape, len(query_ids), len(treatment_ids))

    # log transform input rows, output rows and execution time
    if log_transform:
        df_hash["input_rows"] = np.log10(df_hash["input_rows"]+1)
        df_hash["output_rows"] = np.log10(df_hash["output_rows"]+1)
        df_hash["execution_time"] = np.log10(df_hash["execution_time"]+1)

    df_sampled = df_hash.copy()
    # filter those rows with query_treatment_ids
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    features = ['query_id', 'treatment_id', 'input_rows', 'output_rows', 'execution_time']
    df_hash = df_hash[features]
    df_obs = df_obs[features]
    df_cf = df_cf[features]
    # reset index
    # drop duplicates
    # df_hash = df_hash.drop_duplicates()
    # df_sampled = df_sampled.drop_duplicates()
    df_hash = df_hash.reset_index(drop=True)
    df_obs = df_obs.reset_index(drop=True)
    df_cf = df_cf.reset_index(drop=True)

    # save df_obs and df_cf
    obs_dir = "/Users/ppruthi/research/novelty_accommodation/synthetic_modeling/real_world_data/mathso/obs_data/"
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    df_obs.to_csv(obs_dir + "hash_sampled.csv", index=False)
    df_cf.to_csv(obs_dir + "hash_cf.csv", index=False)

    return df_hash, df_obs, df_cf


def get_hash_join_df(treatment_id=None, filters=True, data_folder_name="mathso", data_dir = "all_csvs", function_type="non_linear"):
    
    df = pd.read_csv('{}/{}/{}/all_hash_join_data.csv'.format(DATA_DIR_ROOT, data_folder_name, data_dir))
    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]
    # modelling only hash joins with one loop for now
    if filters:
        df = df[df["Actual Loops"] == 1]
    df.rename(columns = {'left_child_input_actual_rows':'left_output_rows', 'right_child_input_actual_rows': 'right_output_rows',  'Self Shared Read Blocks':'shared_read_blocks', 'Plan Rows': 'plan_rows', 'Join Type': 'join_type'}, inplace = True)
    if "execution_time" not in df.columns:
        df.rename(columns = {'Self Time':'execution_time'}, inplace = True)
    if treatment_id is not None:
        if isinstance(treatment_id, list):
            df = df[df["treatment_id"].isin(treatment_id)]
        else:
            df = df[df["treatment_id"] == treatment_id]
    
    relevant_features = ['query_id', 'treatment_id','left_output_rows', 'right_output_rows', 'plan_rows', 'join_type', 'output_rows', 'execution_time']
    
    relevant_features += [ 'run_id', 'dbname', 'data_folder_name']
    df_hash_join = df[relevant_features]

    return df_hash_join

def get_hash_join_train_test_split( train_query_ids = None, test_query_ids = None, treatment_id=None, log_transform=True, function_type="non_linear", plan_rows=False, noise_variance=0.0, data_folder_name="mathso", data_dir = "all_csvs"):
    df_hash_join = get_hash_join_df( treatment_id=treatment_id, function_type=function_type, data_folder_name=data_folder_name, data_dir=data_dir)
    if df_hash_join is None:
        return None, None, None, None, None, None, None
    lr, rr, pr, join_type, or_, et, df_hash_join = preprocess_hash_join_data(df_hash_join, log_transform=log_transform)
    # reset index
    df_hash_join = df_hash_join.reset_index(drop=True)
    if train_query_ids is not None and test_query_ids is not None:
        tr_idx = df_hash_join[df_hash_join["query_id"].isin(train_query_ids)].index
        te_idx = df_hash_join[df_hash_join["query_id"].isin(test_query_ids)].index
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df_hash_join)), test_size=0.2, random_state=42)
    lr_tr, rr_tr, pr_tr, jt_tr, or_tr, et_tr = lr[tr_idx], rr[tr_idx], pr[tr_idx], join_type[tr_idx], or_[tr_idx], et[tr_idx]
    lr_te, rr_te, pr_te, jt_te, or_te, et_te = lr[te_idx], rr[te_idx], pr[te_idx], join_type[te_idx], or_[te_idx], et[te_idx]
    train_ids = df_hash_join.iloc[tr_idx]["query_id"].values
    test_ids = df_hash_join.iloc[te_idx]["query_id"].values
    train_treatment_ids = df_hash_join.iloc[tr_idx]["treatment_id"].values
    test_treatment_ids = df_hash_join.iloc[te_idx]["treatment_id"].values

    # train_features = ['left_output_rows', 'right_output_rows', 'output_rows']
    pr_tr_std = or_tr + noise_variance * (pr_tr - or_tr)
    pr_te_std = or_te + noise_variance * (pr_te - or_te)
    if plan_rows:
        X_train = np.concatenate((lr_tr, rr_tr, pr_tr), axis=1)
        X_test = np.concatenate((lr_te, rr_te, pr_te), axis=1)
    else:
        X_train = np.concatenate((lr_tr, rr_tr, or_tr), axis=1)
        X_test = np.concatenate((lr_te, rr_te, or_te), axis=1)

    y_train = et_tr
    y_test = et_te

    # scale data
    if len(X_train) > 0 and len(X_test) > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # y_train = scaler.fit_transform(y_train)
        # y_test = scaler.transform(y_test)
    

        # convert to torch tensor
        X_train = torch.tensor(X_train, dtype=torch.float)
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
    all_indices = np.concatenate([tr_idx, te_idx])
    df_hash_join = df_hash_join.iloc[all_indices]
    return X_train, X_test, y_train, y_test, df_hash_join, train_ids, test_ids, train_treatment_ids, test_treatment_ids

def get_hash_join_obs_df( query_treatment_ids = None, filters=True, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs"):
    # get all treatment ids
    df_hash_join = get_hash_join_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)

    # filter treatment ids and query ids
    df_hash_join = df_hash_join[df_hash_join["query_id"].isin(query_ids)]
    df_hash_join = df_hash_join[df_hash_join["treatment_id"].isin(treatment_ids)]

    # # get those query ids that have atleast two treatments
    # query_ids = df_hash_join.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    # # filter treatment ids and query ids
    # df_hash_join = df_hash_join[df_hash_join["query_id"].isin(query_ids)]
    # # reset index
    df_hash_join = df_hash_join.reset_index(drop=True)
    print(df_hash_join.shape, len(query_ids), len(treatment_ids))

    # log transform input rows, output rows and execution time
    if log_transform:
        df_hash_join["left_output_rows"] = np.log10(df_hash_join["left_output_rows"]+1)
        df_hash_join["right_output_rows"] = np.log10(df_hash_join["right_output_rows"]+1)
        df_hash_join["output_rows"] = np.log10(df_hash_join["output_rows"]+1)
        df_hash_join["execution_time"] = np.log10(df_hash_join["execution_time"]+1)

    df_sampled = df_hash_join.copy()
    # filter those rows with query_treatment_ids
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    features = ['query_id', 'treatment_id', 'left_output_rows', 'right_output_rows', 'output_rows', 'execution_time']
    df_hash_join = df_hash_join[features]
    df_obs = df_obs[features]
    df_cf = df_cf[features]
    # reset index
    # drop duplicates
    # df_hash_join = df_hash_join.drop_duplicates()
    # df_sampled = df_sampled.drop_duplicates()

    df_hash_join = df_hash_join.reset_index(drop=True)
    df_obs = df_obs.reset_index(drop=True)
    df_cf = df_cf.reset_index(drop=True)

    # save df_obs and df_cf
    obs_dir = "/Users/ppruthi/research/novelty_accommodation/synthetic_modeling/real_world_data/mathso/obs_data/"
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    df_obs.to_csv(obs_dir + "hash_join_sampled.csv", index=False)
    df_cf.to_csv(obs_dir + "hash_join_cf.csv", index=False)


    return df_hash_join, df_obs, df_cf

def get_sort_df(treatment_id=None, filters=True, data_folder_name="mathso", data_dir = "all_csvs", function_type="non_linear"):
    
    df = pd.read_csv('{}/{}/{}/all_sort_data.csv'.format(DATA_DIR_ROOT, data_folder_name, data_dir))
    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]
    # modelling only hash joins with one loop for now
    if filters:
        df = df[df["Actual Loops"] == 1]
    df.rename(columns = {'Plan Rows':'plan_rows', 'Plan Rows': 'plan_rows',  'Sort Method': 'sort_method', 'Sort Space Used': 'sort_space_used'}, inplace = True)
    if "execution_time" not in df.columns:
        df.rename(columns = {'Self Time':'execution_time'}, inplace = True)
    if treatment_id is not None:
        if isinstance(treatment_id, list):
            df = df[df["treatment_id"].isin(treatment_id)]
        else:
            df = df[df["treatment_id"] == treatment_id]
    

    relevant_features = ['query_id', 'treatment_id','input_rows',  'plan_rows', 'sort_method',  'output_rows','execution_time']
    
    relevant_features += [ 'run_id', 'dbname', 'data_folder_name']
    
    df_sort = df[relevant_features]
    # reset index
    df_sort = df_sort.reset_index(drop=True)
    return df_sort

def get_sort_train_test_split( train_query_ids = None, test_query_ids = None, treatment_id=None, log_transform=True, function_type="non_linear", plan_rows=False, noise_variance=0.0, data_folder_name="mathso", data_dir = "all_csvs"):
    df_sort = get_sort_df(treatment_id=treatment_id,function_type=function_type,data_folder_name=data_folder_name,data_dir=data_dir)
    if df_sort is None:
        return None, None, None, None, None, None, None
    ir, pr, sort_method, or_, et, df_sort = preprocess_sort_data(df_sort, log_transform=log_transform)
    # reset index
    df_sort = df_sort.reset_index(drop=True)
    if train_query_ids is not None and test_query_ids is not None:
        tr_idx = df_sort[df_sort["query_id"].isin(train_query_ids)].index
        te_idx = df_sort[df_sort["query_id"].isin(test_query_ids)].index
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df_sort)), test_size=0.2, random_state=42)
    ir_tr, pr_tr, sm_tr, or_tr, et_tr = ir[tr_idx], pr[tr_idx], sort_method[tr_idx], or_[tr_idx], et[tr_idx]
    ir_te, pr_te, sm_te, or_te, et_te = ir[te_idx], pr[te_idx], sort_method[te_idx], or_[te_idx], et[te_idx]
    train_ids = df_sort.iloc[tr_idx]["query_id"].values
    test_ids = df_sort.iloc[te_idx]["query_id"].values
    train_treatment_ids = df_sort.iloc[tr_idx]["treatment_id"].values
    test_treatment_ids = df_sort.iloc[te_idx]["treatment_id"].values

    train_features = ['input_rows', 'output_rows', 'sort_method']
    pr_tr_std = or_tr + noise_variance * (pr_tr - or_tr)
    pr_te_std = or_te + noise_variance * (pr_te - or_te)
    if plan_rows:
        X_train = np.concatenate((ir_tr, pr_tr_std), axis=1)
        X_test = np.concatenate((ir_te, pr_te_std), axis=1)
    else:
        X_train = np.concatenate((ir_tr, or_tr), axis=1)
        X_test = np.concatenate((ir_te, or_te), axis=1)
    y_train = et_tr
    y_test = et_te

    # scale data
    if len(X_train) > 0 and len(X_test) > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        

        # convert to torch tensor
        X_train = torch.tensor(X_train, dtype=torch.float)
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
    all_indices = np.concatenate([tr_idx, te_idx])
    df_sort = df_sort.iloc[all_indices]
    return X_train, X_test, y_train, y_test, df_sort, train_ids, test_ids, train_treatment_ids, test_treatment_ids

def get_sort_obs_df( query_treatment_ids = None, filters=True, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs"):
    # get all treatment ids
    df_sort = get_sort_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)

    # filter treatment ids and query ids
    df_sort = df_sort[df_sort["query_id"].isin(query_ids)]
    df_sort = df_sort[df_sort["treatment_id"].isin(treatment_ids)]

    # # get those query ids that have atleast two treatments
    # query_ids = df_sort.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    # # filter treatment ids and query ids
    # df_sort = df_sort[df_sort["query_id"].isin(query_ids)]


    # reset index
    df_sort = df_sort.reset_index(drop=True)
    print(df_sort.shape, len(query_ids), len(treatment_ids))

    # log transform input rows, output rows and execution time
    if log_transform:
        df_sort["input_rows"] = np.log10(df_sort["input_rows"]+1)
        df_sort["output_rows"] = np.log10(df_sort["output_rows"]+1)
        df_sort["execution_time"] = np.log10(df_sort["execution_time"]+1)

    df_sampled = df_sort.copy()
    # filter those rows with query_treatment_ids
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    features = ['query_id', 'treatment_id', 'input_rows', 'output_rows', 'execution_time', 'sort_method']
    df_sort = df_sort[features]
    df_obs = df_obs[features]
    df_cf = df_cf[features]
    # reset index
    # drop duplicates
    # df_sort = df_sort.drop_duplicates()
    # df_sampled = df_sampled.drop_duplicates()
    df_sort = df_sort.reset_index(drop=True)
    df_obs = df_obs.reset_index(drop=True)
    df_cf = df_cf.reset_index(drop=True)

    # save df_obs and df_cf
    obs_dir = "/Users/ppruthi/research/novelty_accommodation/synthetic_modeling/real_world_data/mathso/obs_data/"
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    df_obs.to_csv(obs_dir + "sort_sampled.csv", index=False)
    df_cf.to_csv(obs_dir + "sort_cf.csv", index=False)

    return df_sort, df_obs, df_cf


def get_aggregate_df( treatment_id=None, filters=True, data_folder_name="mathso", data_dir = "all_csvs", function_type="non_linear"):
    
    df = pd.read_csv('{}/{}/{}/all_aggregate_data.csv'.format(DATA_DIR_ROOT, data_folder_name, data_dir))
    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]

    # modelling only hash joins with one loop for now
    if filters:
        df = df[df["Actual Loops"] == 1]
    df.rename(columns = {'Plan Rows':'plan_rows',  'Strategy': 'strategy'}, inplace = True)
    if "execution_time" not in df.columns:
        df.rename(columns = {'Self Time':'execution_time'}, inplace = True)
    if treatment_id is not None:
        if isinstance(treatment_id, list):
            df = df[df["treatment_id"].isin(treatment_id)]
        else:
            df = df[df["treatment_id"] == treatment_id]
   

    relevant_features = ['query_id', 'treatment_id', 'input_rows',  'plan_rows', 'strategy', 'output_rows','execution_time']
    
    relevant_features += ['run_id', 'dbname', 'data_folder_name']
    df_aggregate = df[relevant_features]
    # reset index
    df_aggregate = df_aggregate.reset_index(drop=True)
    return df_aggregate

def get_aggregate_train_test_split( train_query_ids = None, test_query_ids = None, treatment_id=None, log_transform=True, function_type="non_linear", plan_rows=False, noise_variance=0.0, data_folder_name="mathso", data_dir = "all_csvs"):
    df_aggregate = get_aggregate_df(treatment_id=treatment_id, function_type=function_type,data_folder_name=data_folder_name,data_dir=data_dir)
    if df_aggregate is None:
        return None, None, None, None, None, None, None
    ir, pr, strategy, or_, et, df_aggregate = preprocess_aggregate_data(df_aggregate, log_transform=log_transform)
    # reset index
    if train_query_ids is not None and test_query_ids is not None:
        tr_idx = df_aggregate[df_aggregate["query_id"].isin(train_query_ids)].index
        te_idx = df_aggregate[df_aggregate["query_id"].isin(test_query_ids)].index
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df_aggregate)), test_size=0.2, random_state=42)
    ir_tr, pr_tr, st_tr, or_tr, et_tr = ir[tr_idx], pr[tr_idx], strategy[tr_idx], or_[tr_idx], et[tr_idx]
    ir_te, pr_te, st_te, or_te, et_te = ir[te_idx], pr[te_idx], strategy[te_idx], or_[te_idx], et[te_idx]
    train_ids = df_aggregate.iloc[tr_idx]["query_id"].values
    test_ids = df_aggregate.iloc[te_idx]["query_id"].values
    train_treatment_ids = df_aggregate.iloc[tr_idx]["treatment_id"].values
    test_treatment_ids = df_aggregate.iloc[te_idx]["treatment_id"].values

    pr_tr_std = or_tr + noise_variance * (pr_tr - or_tr)
    pr_te_std = or_te + noise_variance * (pr_te - or_te)
    if plan_rows:
        X_train = np.concatenate((ir_tr, pr_tr_std), axis=1)
        X_test = np.concatenate((ir_te, pr_te_std), axis=1)
    else:
        X_train = np.concatenate((ir_tr, or_tr), axis=1)
        X_test = np.concatenate((ir_te, or_te), axis=1)
    y_train = et_tr
    y_test = et_te

    # scale data
    if len(X_train) > 0 and len(X_test) > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # y_train = scaler.fit_transform(y_train)
        # y_test = scaler.transform(y_test)

        # convert to torch tensor
        X_train = torch.tensor(X_train, dtype=torch.float)
        X_test = torch.tensor(X_test, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)
        y_test = torch.tensor(y_test, dtype=torch.float)
    all_indices = np.concatenate([tr_idx, te_idx])
    df_aggregate = df_aggregate.iloc[all_indices]
    return X_train, X_test, y_train, y_test, df_aggregate, train_ids, test_ids, train_treatment_ids, test_treatment_ids

def get_aggregate_obs_df( query_treatment_ids = None, filters=True, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs"):
    # get all treatment ids
    df_aggregate = get_aggregate_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)

    # filter treatment ids and query ids
    df_aggregate = df_aggregate[df_aggregate["query_id"].isin(query_ids)]
    df_aggregate = df_aggregate[df_aggregate["treatment_id"].isin(treatment_ids)]

    # # get those query ids that have atleast two treatments
    # query_ids = df_aggregate.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    # # filter treatment ids and query ids
    # df_aggregate = df_aggregate[df_aggregate["query_id"].isin(query_ids)]

    
    # reset index
    df_aggregate = df_aggregate.reset_index(drop=True)
    print(df_aggregate.shape, len(query_ids), len(treatment_ids))

    # log transform input rows, output rows and execution time
    if log_transform:
        df_aggregate["input_rows"] = np.log10(df_aggregate["input_rows"]+1)
        df_aggregate["output_rows"] = np.log10(df_aggregate["output_rows"]+1)
        df_aggregate["execution_time"] = np.log10(df_aggregate["execution_time"]+1)

    df_sampled = df_aggregate.copy()
    # filter those rows with query_treatment_ids
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    features = ['query_id', 'treatment_id', 'input_rows', 'output_rows', 'execution_time']
    df_aggregate = df_aggregate[features]
    df_obs = df_obs[features]
    df_cf = df_cf[features]
    # reset index
    # drop duplicates
    # df_aggregate = df_aggregate.drop_duplicates()
    # df_sampled = df_sampled.drop_duplicates()
    df_aggregate = df_aggregate.reset_index(drop=True)
    df_obs = df_obs.reset_index(drop=True)
    df_cf = df_cf.reset_index(drop=True)

     # save df_obs and df_cf
    obs_dir = "/Users/ppruthi/research/novelty_accommodation/synthetic_modeling/real_world_data/mathso/obs_data/"
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    df_obs.to_csv(obs_dir + "agg_sampled.csv", index=False)
    df_cf.to_csv(obs_dir + "agg_cf.csv", index=False)

    return df_aggregate, df_obs, df_cf


def get_index_df( treatment_id=None, filters=True, data_folder_name="mathso", data_dir = "all_csvs", function_type="non_linear"):
    
    df = pd.read_csv('{}/{}/{}/all_index_scan_data.csv'.format(DATA_DIR_ROOT, data_folder_name, data_dir))

    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]
    df["Self Shared Read Blocks"] = df["Self Shared Read Blocks"] * df["Actual Loops"]
    if filters:
        df = df[df["Actual Loops"] == 1]
    df.rename(columns = {'Self Shared Read Blocks':'shared_read_blocks', 'Plan Rows': 'plan_rows'}, inplace = True)
    df["filter_query"] = ~df["Filter"].isnull()
    if treatment_id is not None:
        if isinstance(treatment_id, list):
            df = df[df["treatment_id"].isin(treatment_id)]
        else:
            df = df[df["treatment_id"] == treatment_id]
    
    if filters:
        df = df[df["shared_read_blocks"] > 0]
    
    # rename Self Shared Read Blocks to shared_read_blocks and plan rows to plan_rows

    # relevant_features = ['query_id', 'treatment_id','input_rows', 'plan_rows', 'limit_query', 'filter_query', 'output_rows', 'shared_read_blocks', 'execution_time']
    relevant_features = ['query_id', 'treatment_id','input_rows', 'plan_rows', 'output_rows', 'shared_read_blocks', 'execution_time']
    
    relevant_features += [ 'run_id', 'dbname', 'data_folder_name']
    df_index = df[relevant_features]
    df_index = df_index.dropna()
    # reset index
    df_index = df_index.reset_index(drop=True)
    return df_index

def get_index_train_test_split( train_query_ids = None, test_query_ids = None, treatment_id=None, log_transform=True, function_type="non_linear", plan_rows=False, noise_variance=0.1, data_dir="all_csvs", data_folder_name="mathso"):
    df_index = get_index_df(treatment_id=treatment_id, function_type=function_type,data_dir=data_dir, data_folder_name = data_folder_name)
    if df_index is None:
        return None, None, None, None, None, None, None

    # preprocess data
    ir, pr, or_, srb, et, selectivity, df_index = preprocess_index_scan_data(df_index, log_transform=log_transform)
    # reset index
    df_index = df_index.reset_index(drop=True)
    # print(ir.shape, pr.shape, lq.shape, fc.shape, or_.shape, srb.shape, et.shape, selectivity.shape)
    # drop na
    
    if train_query_ids is not None and test_query_ids is not None:
        tr_idx = df_index[df_index["query_id"].isin(train_query_ids)].index
        te_idx = df_index[df_index["query_id"].isin(test_query_ids)].index
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df_index)), test_size=0.2, random_state=42)
    
    ir_tr, pr_tr,  or_tr, srb_tr, et_tr, sel_tr = ir[tr_idx], pr[tr_idx],  or_[tr_idx], srb[tr_idx], et[tr_idx], selectivity[tr_idx]
    ir_te, pr_te,  or_te, srb_te, et_te, sel_te = ir[te_idx], pr[te_idx], or_[te_idx], srb[te_idx], et[te_idx], selectivity[te_idx]
    train_ids = df_index.iloc[tr_idx]["query_id"].values
    test_ids = df_index.iloc[te_idx]["query_id"].values
    train_treatment_ids = df_index.iloc[tr_idx]["treatment_id"].values
    test_treatment_ids = df_index.iloc[te_idx]["treatment_id"].values

    # scale data
    print(ir_tr.shape, or_tr.shape, sel_tr.shape)
    train_features = ['input_rows', 'output_rows']
    pr_tr_std = or_tr + noise_variance * (pr_tr - or_tr)
    pr_te_std = or_te + noise_variance * (pr_te - or_te)
    if plan_rows:
        X_train = np.concatenate((ir_tr, pr_tr_std), axis=1)
        X_test = np.concatenate((ir_te, pr_te_std), axis=1)
    else:
        X_train = np.concatenate((ir_tr, or_tr), axis=1)
        X_test = np.concatenate((ir_te, or_te), axis=1)
    y_train = np.concatenate((srb_tr, et_tr), axis=1)
    y_test = np.concatenate((srb_te, et_te), axis=1)

    # convert to torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    all_indices = np.concatenate([tr_idx, te_idx])
    df_index = df_index.iloc[all_indices]
    return X_train, X_test, y_train, y_test, df_index, train_ids, test_ids, train_treatment_ids, test_treatment_ids


def get_index_scan_obs_df(query_treatment_ids = None, filters=False, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs", index_alone = True):
    # get all treatment ids
    df_index_scan = get_index_df(filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    if not index_alone:
        df_seq_scan = get_seq_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
        df_index_scan["scan_type"] = 1
        df_seq_scan["scan_type"] = 0
    
        # combine seq scan and index scan
        df_index_scan = pd.concat([df_index_scan, df_seq_scan], axis=0)
     # filter treatment ids and query ids
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)
    df_index_scan = df_index_scan[df_index_scan["query_id"].isin(query_ids)]
    df_index_scan = df_index_scan[df_index_scan["treatment_id"].isin(treatment_ids)]
    # sort by query id and treatment id
    df_index_scan = df_index_scan.sort_values(["query_id", "treatment_id"])
    print(df_index_scan[["query_id", "treatment_id"]].head())
    print(query_treatment_ids.head())
    
    # reset index
    df_index_scan = df_index_scan.reset_index(drop=True)
    print(df_index_scan.shape, len(query_ids), len(treatment_ids))

    # log transform input rows, output rows and execution time
    if log_transform:
        df_index_scan["input_rows"] = np.log10(df_index_scan["input_rows"]+1)
        df_index_scan["output_rows"] = np.log10(df_index_scan["output_rows"]+1)
        df_index_scan["execution_time"] = np.log10(df_index_scan["execution_time"]+1)

    df_sampled = df_index_scan.copy()
    # filter those rows with query_treatment_ids
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    features = ['query_id', 'treatment_id', 'input_rows', 'output_rows', 'execution_time']
    if not index_alone:
        features += ["scan_type"]
    df_index_scan = df_index_scan[features]
    df_obs = df_obs[features]
    df_cf = df_cf[features]
    # reset index
    # drop duplicates
    # df_index_scan = df_index_scan.drop_duplicates()
    # df_sampled = df_sampled.drop_duplicates()
    df_index_scan = df_index_scan.reset_index(drop=True)
    df_obs = df_obs.reset_index(drop=True)
    df_cf = df_cf.reset_index(drop=True)

    # save df_obs and df_cf
    obs_dir = "/Users/ppruthi/research/novelty_accommodation/synthetic_modeling/real_world_data/mathso/obs_data/"
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    df_obs.to_csv(obs_dir + "index_sampled.csv", index=False)
    df_cf.to_csv(obs_dir + "index_cf.csv", index=False)

    return df_index_scan, df_obs, df_cf

def get_seq_df( treatment_id=None, filters=True, data_folder_name="mathso", data_dir = "all_csvs", function_type="non_linear"):
    
    print(DATA_DIR_ROOT, data_folder_name, data_dir)
    df = pd.read_csv('{}/{}/{}/all_seq_scan_data.csv'.format(DATA_DIR_ROOT, data_folder_name, data_dir))
    
    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]
    
    # df["Self Shared Read Blocks"] = df["Self Shared Read Blocks"] * df["Actual Loops"]
    if filters:
        df = df[df["Actual Loops"] == 1]
    df.rename(columns = {'Self Shared Read Blocks':'shared_read_blocks', 'Plan Rows': 'plan_rows'}, inplace = True)
    df["filter_query"] = ~df["Filter"].isnull()
    if treatment_id is not None:
        if isinstance(treatment_id, list):
            df = df[df["treatment_id"].isin(treatment_id)]
        else:
            df = df[df["treatment_id"] == treatment_id]
    
    if filters:
        df = df[df["input_rows"] > 0]
        df = df[df["shared_read_blocks"] > 0]
    
    
    df = df.reset_index(drop=True)
    # rename Self Shared Read Blocks to shared_read_blocks and plan rows to plan_rows

    # relevant_features = ['query_id', 'treatment_id','input_rows', 'plan_rows', 'limit_query', 'filter_query', 'output_rows', 'shared_read_blocks', 'execution_time']
    relevant_features = ['query_id', 'treatment_id','input_rows', 'plan_rows', 'output_rows', 'shared_read_blocks', 'execution_time']
    
    relevant_features += [ 'run_id', 'dbname', 'data_folder_name']
    df_seq = df[relevant_features]
    
    df_seq = df_seq.dropna()
    print(len(df_seq))
    # reset index
    return df_seq

def get_seq_train_test_split( train_query_ids = None, test_query_ids = None, treatment_id=None, log_transform=True, function_type="non_linear", plan_rows=False, noise_variance=0.0,data_dir = "all_csvs",data_folder_name="mathso"):
    df_seq = get_seq_df( treatment_id=treatment_id, function_type=function_type, data_dir=data_dir,data_folder_name=data_folder_name)
    # preprocess data
    ir, pr,  or_, srb, et, selectivity, df_seq = preprocess_seq_scan_data(df_seq, log_transform=log_transform)
    # reset index
    df_seq = df_seq.reset_index(drop=True)

    if train_query_ids is not None and test_query_ids is not None:
        tr_idx = df_seq[df_seq["query_id"].isin(train_query_ids)].index
        te_idx = df_seq[df_seq["query_id"].isin(test_query_ids)].index
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df_seq)), test_size=0.2, random_state=42)
    
    ir_tr, pr_tr,  or_tr, srb_tr, et_tr, sel_tr = ir[tr_idx], pr[tr_idx],  or_[tr_idx], srb[tr_idx], et[tr_idx], selectivity[tr_idx]
    ir_te, pr_te,  or_te, srb_te, et_te, sel_te = ir[te_idx], pr[te_idx],  or_[te_idx], srb[te_idx], et[te_idx], selectivity[te_idx]
    train_ids = df_seq.iloc[tr_idx]["query_id"].values
    test_ids = df_seq.iloc[te_idx]["query_id"].values
    train_treatment_ids = df_seq.iloc[tr_idx]["treatment_id"].values
    test_treatment_ids = df_seq.iloc[te_idx]["treatment_id"].values
    # scale data
    train_features = ['input_rows', 'output_rows']
    # add noise variance to plan_rows such that 0 means output_rows and 1 means plan_rows
    pr_tr_std = or_tr + noise_variance * (pr_tr - or_tr)
    pr_te_std = or_te + noise_variance * (pr_te - or_te)
    if plan_rows:
        X_train = np.concatenate((ir_tr, pr_tr_std), axis=1)
        X_test = np.concatenate((ir_te, pr_te_std), axis=1)
    else:
        X_train = np.concatenate((ir_tr, or_tr), axis=1)
        X_test = np.concatenate((ir_te, or_te), axis=1)

    y_train = np.concatenate((srb_tr, et_tr), axis=1)
    y_test = np.concatenate((srb_te, et_te), axis=1)

    # convert to torch tensor
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    all_indices = np.concatenate([tr_idx, te_idx])
    df_seq = df_seq.iloc[all_indices]
   
    return X_train, X_test, y_train, y_test, df_seq, train_ids, test_ids, train_treatment_ids, test_treatment_ids

def get_seq_obs_df( query_treatment_ids = None, filters=True, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs"):
    # get all treatment ids
    # print(query_treatment_ids)
    df_seq = get_seq_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    query_ids = query_treatment_ids["query_id"].unique()
    treatment_ids = query_treatment_ids["treatment_id"].unique()
    query_treatment_ids["query_id_treatment_id"] = query_treatment_ids["query_id"].astype(str) + "_" + query_treatment_ids["treatment_id"].astype(str)
    # print(df_seq[['query_id', 'treatment_id']].head(10))
    # filter treatment ids and query ids
    df_seq = df_seq[df_seq["query_id"].isin(query_ids)]
    df_seq = df_seq[df_seq["treatment_id"].isin(treatment_ids)]
    

    # # get those query ids that have atleast two treatments
    # query_ids = df_seq.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    # # filter treatment ids and query ids
    # df_seq = df_seq[df_seq["query_id"].isin(query_ids)]

    # reset index
    df_seq = df_seq.reset_index(drop=True)
    print(df_seq.shape, len(query_ids), len(treatment_ids))

    # log transform input rows, output rows and execution time
    if log_transform:
        df_seq["input_rows"] = np.log10(df_seq["input_rows"]+1)
        df_seq["output_rows"] = np.log10(df_seq["output_rows"]+1)
        df_seq["execution_time"] = np.log10(df_seq["execution_time"]+1)

    df_sampled = df_seq.copy()
    # filter those rows with query_treatment_ids
    df_sampled["query_id_treatment_id"] = df_sampled["query_id"].astype(str) + "_" + df_sampled["treatment_id"].astype(str)
    df_obs = df_sampled[df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    df_cf = df_sampled[~df_sampled["query_id_treatment_id"].isin(query_treatment_ids["query_id_treatment_id"])]
    features = ['query_id', 'treatment_id', 'input_rows', 'output_rows', 'execution_time']
    df_seq = df_seq[features]
    df_obs = df_obs[features]
    df_cf = df_cf[features]
    # reset index
    # drop duplicates
    # df_seq = df_seq.drop_duplicates()
    # df_sampled = df_sampled.drop_duplicates()
    df_seq = df_seq.reset_index(drop=True)
    df_obs = df_obs.reset_index(drop=True)
    df_cf = df_cf.reset_index(drop=True)

    # save df_obs and df_cf
    obs_dir = "/Users/ppruthi/research/novelty_accommodation/synthetic_modeling/real_world_data/mathso/obs_data/"
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    df_obs.to_csv(obs_dir + "seq_sampled.csv", index=False)
    df_cf.to_csv(obs_dir + "seq_cf.csv", index=False)

    return df_seq, df_obs, df_cf

def generate_high_level_real_world_data(treatment_ids=None, data_folder_name="mathso", data_dir = "all_csvs", plan_rows=False, noise_variance=0.0, filters=False):
    df_seq = get_seq_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    if df_seq is None:
        df_seq_agg = None
    else:
        num_df_seq = df_seq.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).count()
        if plan_rows:
            df_seq["plan_rows_std"] = np.log10(df_seq["output_rows"]+1) + noise_variance * (np.log10(df_seq["plan_rows"]+1) - np.log10(df_seq["output_rows"]+1))
            df_seq["plan_rows_std"] = np.power(10, df_seq["plan_rows_std"]) - 1
        else:
            df_seq["plan_rows_std"] = df_seq["plan_rows"]
        df_seq_agg = df_seq.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).agg({"input_rows": "sum", "output_rows": "sum", "plan_rows":"sum", "plan_rows_std": "sum",  "shared_read_blocks": "sum", "execution_time": "sum"})
        df_seq_agg["num_Seq Scan"] = num_df_seq["input_rows"]
        df_seq_agg = df_seq_agg.reset_index()
        renamed_colnames = ["input_rows", "output_rows", "plan_rows", "plan_rows_std", "shared_read_blocks", "execution_time"]
        # add prefix to column names
        df_seq_agg = df_seq_agg.rename(columns = {col: "Seq Scan_" + col for col in renamed_colnames})

    df_index = get_index_df(filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    if df_index is None:
        df_index_agg = None
    else:
        num_df_index = df_index.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).count()
        if plan_rows:
            df_index["plan_rows_std"] = np.log10(df_index["output_rows"]+1) + noise_variance * (np.log10(df_index["plan_rows"]+1) - np.log10(df_index["output_rows"]+1))
            df_index["plan_rows_std"] = np.power(10, df_index["plan_rows_std"]) - 1
        else:
            df_index["plan_rows_std"] = df_index["plan_rows"]
        df_index_agg = df_index.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).agg({"input_rows": "sum", "output_rows": "sum", "plan_rows":"sum", "plan_rows_std": "sum", "shared_read_blocks": "sum", "execution_time": "sum"})
        df_index_agg["num_Index Scan"] = num_df_index["input_rows"]
        df_index_agg = df_index_agg.reset_index()
        renamed_colnames = ["input_rows", "output_rows", "plan_rows","plan_rows_std", "shared_read_blocks", "execution_time"]
        # add prefix to column names
        df_index_agg = df_index_agg.rename(columns = {col: "Index Scan_" + col for col in renamed_colnames})

    df_hash = get_hash_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    # print(df_hash.head(10))

    if df_hash is None:
        df_hash_agg = None
    else:
        
        num_df_hash = df_hash.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).count()
        if plan_rows:
            df_hash["plan_rows_std"] = np.log10(df_hash["output_rows"]+1) + noise_variance * (np.log10(df_hash["plan_rows"]+1) - np.log10(df_hash["output_rows"]+1))
            df_hash["plan_rows_std"] = np.power(10, df_hash["plan_rows_std"]) - 1
        else:
            df_hash["plan_rows_std"] = df_hash["plan_rows"]
        df_hash_agg = df_hash.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).agg({"input_rows": "sum", "output_rows": "sum", "plan_rows":"sum",  "execution_time": "sum", "hash_buckets": "sum", "plan_rows_std": "sum"})
        df_hash_agg["num_Hash"] = num_df_hash["input_rows"]
        df_hash_agg = df_hash_agg.reset_index()
        df_hash_agg["hash_buckets"] = np.log2(df_hash_agg["hash_buckets"]+1)
        renamed_colnames = ["input_rows", "output_rows", "plan_rows", "execution_time", "hash_buckets", "plan_rows_std"]
        # add prefix to column names
        df_hash_agg = df_hash_agg.rename(columns = {col: "Hash_" + col for col in renamed_colnames})
        

    df_hash_join = get_hash_join_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    if df_hash_join is None:
        df_hash_join_agg = None
    else:
        num_df_hash_join = df_hash_join.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).count()
        if plan_rows:
            df_hash_join["plan_rows_std"] = np.log10(df_hash_join["output_rows"]+1) + noise_variance * (np.log10(df_hash_join["plan_rows"]+1) - np.log10(df_hash_join["output_rows"]+1))
            df_hash_join["plan_rows_std"] = np.power(10, df_hash_join["plan_rows_std"]) - 1
        else:
            df_hash_join["plan_rows_std"] = df_hash_join["plan_rows"]
        df_hash_join_agg = df_hash_join.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).agg({"left_output_rows": "sum", "right_output_rows": "sum", "output_rows": "sum", "plan_rows":"sum", "execution_time": "sum", "plan_rows_std": "sum"})
        df_hash_join_agg["num_Hash Join"] = num_df_hash_join["left_output_rows"]
        df_hash_join_agg = df_hash_join_agg.reset_index()
        renamed_colnames = ["left_output_rows", "right_output_rows", "output_rows", "plan_rows", "execution_time", "plan_rows_std"]
        # add prefix to column names
        df_hash_join_agg = df_hash_join_agg.rename(columns = {col: "Hash Join_" + col for col in renamed_colnames})

    df_sort = get_sort_df(filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)
    
    if df_sort is None:
        df_sort_agg = None
    else:
        num_df_sort = df_sort.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).count()
        if plan_rows:
            df_sort["plan_rows_std"] = np.log10(df_sort["output_rows"]+1) + noise_variance * (np.log10(df_sort["plan_rows"]+1) - np.log10(df_sort["output_rows"]+1))
            df_sort["plan_rows_std"] = np.power(10, df_sort["plan_rows_std"]) - 1
        else:
            df_sort["plan_rows_std"] = df_sort["plan_rows"]
        df_sort_agg = df_sort.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).agg({"input_rows": "sum", "output_rows": "sum", "plan_rows":"sum", "execution_time": "sum", "sort_method": "first", "plan_rows_std": "sum"})
        df_sort_agg["num_Sort"] = num_df_sort["input_rows"]
        df_sort_agg = df_sort_agg.reset_index()
        df_sort_agg["sort_method"] = df_sort_agg["sort_method"].fillna("top-N heapsort")
        sort_method_pd = df_sort_agg["sort_method"].values.reshape(-1,1)
        # convert to categorical
        # replace null with "top-N heapsort"
        # replace None with "top-N heapsort"
        sort_method_unique = np.unique(sort_method_pd)
        sort_method_dict = {sort_method_unique[i]: i for i in range(len(sort_method_unique))}
        
        sort_method = []
        for sm in sort_method_pd:
            try:
                sort_method.append(sort_method_dict[sm])
            except:
                sort_method.append(sort_method_dict[sm[0]])
        sort_method = np.array(sort_method).reshape(-1,1)
        df_sort_agg["sort_method"] = sort_method
        
        renamed_colnames = ["input_rows", "output_rows", "plan_rows", "sort_method", "execution_time", "plan_rows_std"]
        # add prefix to column names
        df_sort_agg = df_sort_agg.rename(columns = {col: "Sort_" + col for col in renamed_colnames})

    df_aggregate = get_aggregate_df( filters=filters, data_folder_name=data_folder_name, data_dir=data_dir)

    if df_aggregate is None:
        df_aggregate_agg = None
    else:
        num_df_aggregate = df_aggregate.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).count()
        if plan_rows:
            df_aggregate["plan_rows_std"] = np.log10(df_aggregate["output_rows"]+1) + noise_variance * (np.log10(df_aggregate["plan_rows"]+1) - np.log10(df_aggregate["output_rows"]+1))
            df_aggregate["plan_rows_std"] = np.power(10, df_aggregate["plan_rows_std"]) - 1
        else:
            df_aggregate["plan_rows_std"] = df_aggregate["plan_rows"]
        df_aggregate_agg = df_aggregate.groupby(["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"]).agg({"input_rows": "sum", "output_rows": "sum", "plan_rows":"sum", "execution_time": "sum", "strategy": "first", "plan_rows_std": "sum"})
        df_aggregate_agg["num_Aggregate"] = num_df_aggregate["input_rows"]
        df_aggregate_agg = df_aggregate_agg.reset_index()
        df_aggregate_agg["strategy"] = df_aggregate_agg["strategy"].fillna("Sorted")
        
        strategy_pd = df_aggregate_agg["strategy"].values.reshape(-1,1)
        # convert to categorical
        # replace null with "Sorted"
        
        strategy_unique = np.unique(strategy_pd)
        strategy_unique_dict = {strategy_unique[i]: i for i in range(len(strategy_unique))}
        
        aggregate_strategy = []
        for st in strategy_pd:
            try:
                aggregate_strategy.append(strategy_unique_dict[st])
            except:
                aggregate_strategy.append(strategy_unique_dict[st[0]])
        aggregate_strategy = np.array(aggregate_strategy).reshape(-1,1)
        df_aggregate_agg["strategy"] = aggregate_strategy
        renamed_colnames = ["input_rows", "output_rows", "plan_rows", "strategy", "execution_time", "plan_rows_std"]

        # add prefix to column names
        df_aggregate_agg = df_aggregate_agg.rename(columns = {col: "Aggregate_" + col for col in renamed_colnames})

    if df_index_agg is not None:
        df = df_seq_agg.merge(df_index_agg, on=["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"], how="outer")
    else:
        df = df_seq_agg

    if df_hash_agg is not None:
        df = df.merge(df_hash_agg, on=["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"], how="outer")
    if df_hash_join_agg is not None:
        df = df.merge(df_hash_join_agg, on=["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"], how="outer")
    if df_sort_agg is not None:
        df = df.merge(df_sort_agg, on=["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"], how="outer")
    if df_aggregate_agg is not None:
        df = df.merge(df_aggregate_agg, on=["query_id", "treatment_id", "run_id", "dbname", "data_folder_name"], how="outer")

    # filter only certain query ids
    main_research_dir = "/Users/ppruthi/research/novelty_accommodation/"
    modeling_dir = os.path.join(main_research_dir, "modeling")
    query_id_json_path = "{}/old_models_results/jsons/query_ids_dict.json".format(modeling_dir)
    query_ids_schema = json.load(open(query_id_json_path, "r"))

    
    
    
    df = df[(df["data_folder_name"] == data_folder_name)]
    df = df[df["run_id"] == 0]
    df = df[df["query_id"].isin(query_ids_schema["all_query_ids"])]
    

    if treatment_ids is not None:
        if isinstance(treatment_ids, list):
            df = df[df["treatment_id"].isin(treatment_ids)]
        else:
            df = df[df["treatment_id"] == treatment_ids]
        

    # reset index
    df = df.reset_index(drop=True)

    return df

def preprocess_high_level_data(treatment_id=None, data_folder_name="mathso", log_transform=True, data_dir = "all_csvs", function_type="non_linear", plan_rows=False, noise_variance=0.0, return_scm_structure=False, train_query_ids = None, test_query_ids = None, return_full_structure=False, return_individual_et = True, filters=False):
    print("Preprocessing high level data")
    print(DATA_DIR_ROOT, data_folder_name, data_dir)
    df_orig = generate_high_level_real_world_data(treatment_ids=treatment_id, data_folder_name=data_folder_name, data_dir=data_dir,plan_rows=plan_rows, noise_variance=noise_variance, filters=filters)

    df_orig = df_orig.fillna(0)

    
    df_orig = df_orig.reset_index(drop = True)

    # fill na with 0
    df = df_orig.copy()
    
    if "num_Hash Join" not in df.columns:
        df["num_Hash Join"] = 0
    if "num_Sort" not in df.columns:
        df["num_Sort"] = 0
    if "num_Aggregate" not in df.columns:
        df["num_Aggregate"] = 0
    df["num_complex_ops"] = df["num_Sort"].astype(bool).astype(int) + df["num_Aggregate"].astype(bool).astype(int) + df["num_Hash Join"].astype(bool).astype(int)
    
    
    # relevant_features = ['input_rows', 'plan_rows', 'limit_query', 'filter_query', 'output_rows', 'shared_read_blocks', 'execution_time']
    
    rows_features = []
    num_features = []
    categorical_features = []
    srb_output_features = []
    et_output_features = []
    total_features = ["total_shared_read_blocks", "total_execution_time"]
    for col in df.columns:
        if col.startswith("num"):
            num_features.append(col)
        
        if plan_rows:
            if col.__contains__("rows_std") and not col.__contains__("sibling") and not col.__contains__("output"):
                rows_features.append(col)
            if col.__contains__("left_output_rows") and not col.__contains__("sibling") and not col.__contains__("plan"):
                rows_features.append(col)
            if col.__contains__("right_output_rows") and not col.__contains__("sibling") and not col.__contains__("plan"):
                rows_features.append(col)
            if col.__contains__("input_rows") and not col.__contains__("sibling") and not col.__contains__("plan"):
                rows_features.append(col)
        else:
            if col.__contains__("rows") and not col.__contains__("sibling") and not col.__contains__("plan"):
                rows_features.append(col)
        if col.__contains__("time"):
            et_output_features.append(col)
        if col.__contains__("blocks") and (col.__contains__("Seq Scan") or col.__contains__("Index Scan")):
            srb_output_features.append(col)
        if treatment_id != None and isinstance(treatment_id, list) and len(treatment_id) > 1:
            if col.__contains__("method"):
                categorical_features.append(col)
            if col.__contains__("strategy"):
                categorical_features.append(col)
            if col.__contains__("buckets"):
                categorical_features.append(col)
                
        
   
    print(rows_features)
   
    # print("train features: {}".format(rows_features + num_features))
    # print("output features: {}".format(srb_output_features + et_output_features))
    df = df[["query_id", "treatment_id"] + num_features + rows_features + categorical_features + srb_output_features + et_output_features]
    df["total_shared_read_blocks"] = df[srb_output_features].sum(axis=1)
    df["total_execution_time"] = df[et_output_features].sum(axis=1)
   
    preprocess_df = df.copy()

    # log transform row_features, srb_output_features, et_output_features
    if log_transform:
        preprocess_df[rows_features] = np.log10(preprocess_df[rows_features] + 1)
        preprocess_df[srb_output_features] = np.log10(preprocess_df[srb_output_features] + 1)
        preprocess_df[et_output_features] = np.log10(preprocess_df[et_output_features] + 1)

        preprocess_df["total_shared_read_blocks"] = np.log10(preprocess_df["total_shared_read_blocks"] + 1)
        preprocess_df["total_execution_time"] = np.log10(preprocess_df["total_execution_time"] + 1)
    # print(preprocess_df[preprocess_df["query_id"] == 51439].iloc[0])
    # save preprocess_df
    preprocess_df.to_csv("{}/{}/all_csvs/query_plan_high_level_features.csv".format(DATA_DIR_ROOT, data_folder_name), index=False)
    X = preprocess_df[num_features + rows_features + categorical_features]
    # drop columns with only 1 unique value
    # if not an instance of list
    if treatment_id != None and isinstance(treatment_id, list) and len(treatment_id) > 1:
        if 0 in treatment_id:
            preprocess_df = preprocess_df.loc[:, preprocess_df.apply(pd.Series.nunique) != 1]
            X = X.loc[:, X.apply(pd.Series.nunique) != 1]
    else:
        if treatment_id == 0:
            preprocess_df = preprocess_df.loc[:, preprocess_df.apply(pd.Series.nunique) != 1]
            X = X.loc[:, X.apply(pd.Series.nunique) != 1]
        
    if "Sort_sort_method" in X.columns:
        sort_method_pd = X["Sort_sort_method"].values.reshape(-1,1)
        # convert to categorical
        # replace 0 with None
        # get unique without 0
        try:
            sort_method_pd[sort_method_pd == 0] = "None"
        except:
            print("did not find 0")
        sort_method_unique = np.unique(sort_method_pd)
        # remove None
        sort_method_dict = {sort_method_unique[i]: i for i in range(len(sort_method_unique))}
        sort_method = []
        for sm in sort_method_pd:
            try:
                sort_method.append(sort_method_dict[sm])
            except:
                sort_method.append(sort_method_dict[sm[0]])
        sort_method = np.array(sort_method).reshape(-1,1)
        X["Sort_sort_method"] = sort_method

    if "Aggregate_strategy" in X.columns:
        aggregate_strategy_pd = X["Aggregate_strategy"].values.reshape(-1,1)
        try:
            aggregate_strategy_pd[aggregate_strategy_pd == 0] = "None"
        except:
            print("did not find 0")

        # convert to categorical
        aggregate_strategy_unique = np.unique(aggregate_strategy_pd)
        aggregate_strategy_dict = {aggregate_strategy_unique[i]: i for i in range(len(aggregate_strategy_unique))}
        aggregate_strategy = []
        for sm in aggregate_strategy_pd:
            try:
                aggregate_strategy.append(aggregate_strategy_dict[sm])
            except:
                aggregate_strategy.append(aggregate_strategy_dict[sm[0]])
        aggregate_strategy = np.array(aggregate_strategy).reshape(-1,1)
        X["Aggregate_strategy"] = aggregate_strategy

    y = preprocess_df[total_features].values
    print("train_features", X.columns)
    print("output_features", total_features)
    
    outcome_features = et_output_features
    if "total_execution_time" in outcome_features:
        outcome_features.remove("total_execution_time")

    covariates = X.columns.tolist()
    outcome_features = outcome_features
    for f in covariates:
        if f not in preprocess_df.columns:
            # remove from covariates
            covariates.remove(f)
    for f in outcome_features:
        if f not in preprocess_df.columns:
            # remove from outcome_features
            outcome_features.remove(f)
    print(preprocess_df.columns)
    print("covariates", covariates)
    print("outcome_features", outcome_features)
    if "Aggregate_strategy" in covariates:
        # remove from covariates
        covariates.remove("Aggregate_strategy")
    if "Sort_sort_method" in covariates:
        # remove from covariates
        covariates.remove("Sort_sort_method")
    X = X.values
    if return_scm_structure:
        data_scm = preprocess_df.copy()
        data_scm = data_scm[data_scm["query_id"].isin(train_query_ids)]
        # reset index
        data_scm = data_scm.reset_index(drop=True)
        feature_dict = learn_ges_structure(data_scm, covariates = covariates, outcome_features=outcome_features)
        return preprocess_df, X, y, df_orig, feature_dict
    else:
        if return_full_structure:
            full_feature_dict = {}
            for o in outcome_features:
                full_feature_dict[o] = covariates
            return preprocess_df, X, y, df_orig, full_feature_dict
        else:
            return preprocess_df, X, y, df_orig

def get_high_level_data( train_test_criteria = "random", treatment_id=None, train_size = 1.0, data_folder_name="mathso", log_transform=True, function_type = "non_linear", plan_rows=False, noise_variance=0.0, data_dir = "all_csvs",filters = False):
    df, X, y, df_orig = preprocess_high_level_data( treatment_id=treatment_id, data_folder_name=data_folder_name, log_transform=log_transform, function_type=function_type, plan_rows=plan_rows, noise_variance=noise_variance, data_dir = data_dir, filters=filters)
    
    zero_query_ids = df[df["num_complex_ops"] == 0]["query_id"].values
    one_query_ids = df[df["num_complex_ops"] == 1]["query_id"].values
    two_query_ids = df[df["num_complex_ops"] == 2]["query_id"].values
    three_query_ids = df[df["num_complex_ops"] == 3]["query_id"].values
    all_query_ids = list(df["query_id"].unique())
    
    if train_test_criteria == "random":
        train_query_ids, test_query_ids = train_test_split(all_query_ids, test_size=0.2, random_state=42)
        tr_idx = df[df["query_id"].isin(train_query_ids)].index
        te_idx = df[df["query_id"].isin(test_query_ids)].index
    elif train_test_criteria == "test_3":
        train_query_ids = np.concatenate((zero_query_ids, one_query_ids))
        test_query_ids = three_query_ids
        tr_idx = df[df["query_id"].isin(train_query_ids)].index
        te_idx = df[df["query_id"].isin(test_query_ids)].index
    elif train_test_criteria == "test_2":
        train_query_ids = np.concatenate((zero_query_ids, one_query_ids))
        test_query_ids = two_query_ids
        tr_idx = df[df["query_id"].isin(train_query_ids)].index
        te_idx = df[df["query_id"].isin(test_query_ids)].index
    elif train_test_criteria == "test_1":
        train_query_ids = zero_query_ids
        test_query_ids = one_query_ids
        tr_idx = df[df["query_id"].isin(train_query_ids)].index
        te_idx = df[df["query_id"].isin(test_query_ids)].index
    elif train_test_criteria == "test_2_3":
        train_query_ids = np.concatenate((zero_query_ids, one_query_ids))
        test_query_ids = np.concatenate((two_query_ids, three_query_ids))
        tr_idx = df[df["query_id"].isin(train_query_ids)].index
        te_idx = df[df["query_id"].isin(test_query_ids)].index
    elif train_test_criteria == "test_more_2_instances":
        train_query_ids = df[(df["num_Aggregate"] < 1) & (df["num_Sort"] < 1) & (df["num_Hash Join"] < 1)]["query_id"]
        test_query_ids = df[(df["num_Aggregate"] > 1) | (df["num_Sort"] > 1) | (df["num_Hash Join"] > 1)]["query_id"]
        tr_idx = df[df["query_id"].isin(train_query_ids)].index
        te_idx = df[df["query_id"].isin(test_query_ids)].index
    
    if train_size < 1.0:
        training_sample_size = int(train_size * len(tr_idx))
        tr_idx = np.random.choice(tr_idx, size=training_sample_size, replace=False)
    train_query_ids, test_query_ids = df_orig.iloc[tr_idx]["query_id"].values, df_orig.iloc[te_idx]["query_id"].values
    train_treatment_ids, test_treatment_ids = df_orig.iloc[tr_idx]["treatment_id"].values, df_orig.iloc[te_idx]["treatment_id"].values
    X_train, X_test, y_train, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    return X_train, X_test, y_train, y_test, df, train_query_ids, test_query_ids, train_treatment_ids, test_treatment_ids


def generate_novelty_accommodation_datasets_1(treatment_ids = [0, 3], data_folder_name="mathso", 
                                              log_transform=True, plot_folder = None, data_dir = "all_csvs", filters = False):
    print("treatment ids: {}".format(treatment_ids))
    df, X, y, df_orig = preprocess_high_level_data(treatment_id=treatment_ids, data_folder_name=data_folder_name, log_transform=log_transform, data_dir=data_dir, filters=filters)
    df.sort_values(by=["query_id", "treatment_id"], inplace=True)
    df = df[df["treatment_id"].isin(treatment_ids)]
    # df = df[df["num_complex_ops"] > 0]
    df = df.reset_index(drop=True)
    
    # have only query ids that have both treatment ids
    query_ids = df.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    df = df[df["query_id"].isin(query_ids)]
    df = df.reset_index(drop=True)
    return df


def generate_high_level_observational_dataset( treatment_ids = [0, 3], data_folder_name="mathso", sampling = "random", prob_value = 0.5, log_transform=True, biasing_covariate = "Sort_input_rows", bias_strength = 1, plot_folder = None, data_dir = "all_csvs", filters = False):
    print("treatment ids: {}".format(treatment_ids))
    df, X, y, df_orig = preprocess_high_level_data(treatment_id=treatment_ids, data_folder_name=data_folder_name, log_transform=log_transform, data_dir=data_dir, filters=filters)
    df.sort_values(by=["query_id", "treatment_id"], inplace=True)
    print(df["treatment_id"].value_counts())
    # first filter treatment ids and complex ops
    df = df[df["treatment_id"].isin(treatment_ids)]
    # df = df[df["num_complex_ops"] > 0]
    df = df.reset_index(drop=True)
    
    # have only query ids that have both treatment ids
    query_ids = df.groupby("query_id").filter(lambda x: len(x) == len(treatment_ids))["query_id"].unique()
    df = df[df["query_id"].isin(query_ids)]
    df = df.reset_index(drop=True)
    # if treatment_ids.__contains__(6) and real == True:
        # # get query ids that have two different sort methods
        # if "Aggregate_strategy" in df.columns:
        #     agg_query_ids = df.groupby("query_id").filter(lambda x: len(x["Aggregate_strategy"].unique()) > 1)["query_id"].unique()
        # else:
        #     agg_query_ids = []
        
        # if "Sort_sort_method" in df.columns:
        #     sort_query_ids = df.groupby("query_id").filter(lambda x: len(x["Sort_sort_method"].unique()) > 1)["query_id"].unique()
        # else:
        #     sort_query_ids = []
        
        # if "Hash_hash_buckets" in df.columns:
        #     hash_query_ids = []
        #     hash_query_ids = df.groupby("query_id").filter(lambda x: len(x["Hash_hash_buckets"].unique()) > 1)["query_id"].unique()
        # else:
        #     hash_query_ids = []
        
        # union_queries = list(set(agg_query_ids).union(set(sort_query_ids)).union(set(hash_query_ids)))

        
            

    # if len(union_queries) > 0:
    #     df = df[df["query_id"].isin(union_queries)]

    # randomly sample from the treatment ids per query id
    if treatment_ids is not None:
        if sampling == "random":
            df_sampled = df.groupby("query_id").sample(n=1, random_state=42)
            df_cf_sampled = df[~df.index.isin(df_sampled.index)]
        else:
            if sampling == "random_prob":
                df_sampled, df_cf_sampled = random_sampling(df, prob_value=prob_value, treatment_ids=treatment_ids)
            elif sampling == "observational":
                df_sampled, df_cf_sampled = observational_sampling(df, biasing_covariate = biasing_covariate, bias_strength = bias_strength, treatment_ids=treatment_ids, plot_folder=plot_folder)
            
        # random sampling of one of the treatments (no observational bias)
        df_sampled = df_sampled.reset_index(drop=True)
        df_cf_sampled = df_cf_sampled.reset_index(drop=True)
    else:
        df_sampled = df
        df_cf_sampled = None

    # save df_sampled and df_cf_sampled
    obs_dir = "{}/{}/obs_data/".format(DATA_DIR_ROOT, data_folder_name)
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)
    df_sampled.to_csv(f"{obs_dir}/query_plan_high_level_features_sampled.csv", index=False)
    if df_cf_sampled is not None:
        df_cf_sampled.to_csv(f"{obs_dir}/query_plan_high_level_features_cf_sampled.csv", index=False)
    return df, df_sampled, df_cf_sampled


def observational_sampling(df, biasing_covariate = None, bias_strength = 1, treatment_ids = [0, 3], plot_folder = None):
    # sort df by treatment id
    print("DF size: {}".format(df.shape))
    drop = False
    df.sort_values(by=["query_id", "treatment_id"], inplace=True)

    if biasing_covariate == "scan_input_rows":
        if "Index Scan_input_rows" not in df.columns:
            df["Index Scan_input_rows"] = 0
        df["scan_input_rows"] = df["Seq Scan_input_rows"] + df["Index Scan_input_rows"] # + df["Hash_input_rows"] + df["Hash Join_left_output_rows"] + df["Hash Join_right_output_rows"] + df["Sort_input_rows"] + df["Aggregate_input_rows"]
        drop = True
    if biasing_covariate == "scan_output_rows":
        if "Index Scan_output_rows" not in df.columns:
            df["Index Scan_output_rows"] = 0
        df["scan_output_rows"] = df["Seq Scan_output_rows"] + df["Index Scan_output_rows"]
        drop = True
    if biasing_covariate == "selectivity":
        # as rows are log transformed, selectivity is the difference between the log transformed rows
        df["selectivity"] = df["Seq Scan_output_rows"] - df["Seq Scan_input_rows"]
        df["selectivity"] = 10**df["selectivity"]
        drop = True
    if biasing_covariate == "total_output_rows":
        if "Index Scan_output_rows" not in df.columns:
            df["Index Scan_output_rows"] = 0
        df["total_output_rows"] = df["Seq Scan_output_rows"] + df["Index Scan_output_rows"] + df["Hash_output_rows"] + df["Hash Join_output_rows"] + df["Sort_output_rows"] + df["Aggregate_output_rows"]
        drop = True
    if biasing_covariate == "total_input_rows":
        if "Index Scan_input_rows" not in df.columns:
            df["Index Scan_input_rows"] = 0
        df["total_input_rows"] = df["Seq Scan_input_rows"] + df["Index Scan_input_rows"] + df["Hash_input_rows"] + df["Hash Join_left_output_rows"] + df["Hash Join_right_output_rows"] + df["Sort_input_rows"] + df["Aggregate_input_rows"]
        drop = True
        
    cov = df[biasing_covariate].values

    # drop the biasing covariate column
    if drop:
        df = df.drop(columns=[biasing_covariate])
    # get only even rows
    cov = cov[::2]
    ecdf = ECDF(cov)
    cov_ecdf = ecdf(cov)
    cov_ecdf = cov_ecdf - np.mean(cov_ecdf)
    coefficients = np.repeat(bias_strength, len(cov))
    prob_values = 1/(1 + np.exp(-coefficients * cov_ecdf))
    # assign greater than .999 as .999
    prob_values[prob_values > .999] = .999
    # assign less than .001 as .001
    prob_values[prob_values < .001] = .001
    assigned_treatment_ids = np.random.binomial(1, prob_values)
    assigned_treatment_ids = np.where(assigned_treatment_ids == 1, treatment_ids[0], treatment_ids[1])
    # repeat the assigned treatment ids twice
    assigned_treatment_ids = np.repeat(assigned_treatment_ids, 2)
    df["assigned_treatment_id"] = assigned_treatment_ids

    # # plot prob values vs covariate values
    # plt.figure(figsize=(10,10))
    # sns.scatterplot(cov, prob_values)
    # plt.xlabel(biasing_covariate)
    # plt.ylabel("prob_values")
    # plot_dir = "plots/{}/obs_biasing_plots".format(plot_folder)
    # if not os.path.exists(plot_dir):
    #     os.makedirs(plot_dir)
    # plt.savefig("{}/prob_values_vs_covariate_values_bias_strength_{}.png".format(plot_dir, bias_strength))


    # sample rows where treatment id is assigned treatment id
    df_sampled = df[df["treatment_id"] == df["assigned_treatment_id"]]
    df_cf_sampled = df[~df.index.isin(df_sampled.index)]
    return df_sampled, df_cf_sampled

def random_sampling(df, prob_value = 0.5, treatment_ids = [0, 3]):
    query_id_treatment_ids = []
    grouped_df = df.groupby("query_id")

    # based on the covariates, have a biasing function that assigns a probability to each treatment id such that the strength of the bias is adjustable. 

    # if sorting input rows is high or aggregation input rows is high, then choose high memory level with high probability and low memory level with low probability
    # if sorting input rows is low or aggregation input rows is low, then choose high memory level with low probability and low memory level with high probability
    for index, row in grouped_df:
        treatment_id = np.random.choice(treatment_ids, size=1, p=[prob_value, 1 - prob_value])[0]
        query_id_treatment_ids.append([row["query_id"].values[0], treatment_id])
    
    query_id_treatment_ids = pd.DataFrame(query_id_treatment_ids, columns=["query_id", "treatment_id"])
    query_id_treatment_ids["query_id_treatment_id"] = query_id_treatment_ids["query_id"].astype(str) + "_" + query_id_treatment_ids["treatment_id"].astype(str)
    df["query_id_treatment_id"] = df["query_id"].astype(str) + "_" + df["treatment_id"].astype(str)
    df_sampled = df[df["query_id_treatment_id"].isin(query_id_treatment_ids["query_id_treatment_id"])]
    df_cf_sampled = df[~df["query_id_treatment_id"].isin(query_id_treatment_ids["query_id_treatment_id"])]
    return df_sampled, df_cf_sampled
        


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test, preprocess_df, train_query_ids, test_query_ids = get_high_level_data(real = False, train_test_criteria = "random", treatment_id=None, train_size = 1.0)

    df = generate_high_level_observational_dataset(treatment_ids = [0, 6])
