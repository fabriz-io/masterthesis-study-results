# %%

import pandas as pd
import json
import os
import numpy as np

number = "006"

logs_dir_path = f"./log_folder/{number}"
log_files_paths = os.listdir(logs_dir_path)

events = [log_path.split("_")[-1] for log_path in log_files_paths]


df_list = []


for e in np.unique(events):
    event_logs = []

    for log_file in log_files_paths:
        if e in log_file:
            with open(f"{logs_dir_path}/{log_file}") as f:
                log_dict = json.load(f)
            event_logs.append(log_dict)

    df_list.append([e, pd.DataFrame(event_logs)])


df_path = "./dataframes"

os.makedirs(df_path, exist_ok=True)

for df_l in df_list:
    event_name = df_l[0]
    df = df_l[1]

    df.to_csv(f"{df_path}/{event_name}.csv")

# %%
