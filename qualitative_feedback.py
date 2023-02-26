# %%

import pandas as pd

# accepted_

result_df = pd.read_csv("dis_results.csv")
result_df = result_df.drop(columns=["Unnamed: 0"])
accepted_userids = pd.read_csv("accepted_userids.csv")

ui_feedback = (
    pd.read_csv("dataframes/end-textarea-submit.log.csv")
    .sort_values(by="datetime")
    .drop_duplicates(subset=["userid"], keep="last")
)

# result_df = result_df.loc[result_df.userid.isin(accepted_userids.userid)]

ui_feedback_c1 = ui_feedback.text.loc[ui_feedback.condition == "c1"]
ui_feedback_c2 = ui_feedback.text.loc[ui_feedback.condition == "c2"]
ui_feedback_c3 = ui_feedback.text.loc[ui_feedback.condition == "c3"]


# %%

result_df = pd.read_csv("dis_results.csv")
result_df = result_df.drop(columns=["Unnamed: 0"])
accepted_userids = pd.read_csv("accepted_userids.csv")

general_feedback = (
    pd.read_csv("dataframes/textarea-general-feedback.log.csv")
    .sort_values(by="datetime")
    .drop_duplicates(subset=["userid"], keep="last")
    .dropna(subset=["text"])
)

# result_df = result_df.loc[result_df.userid.isin(accepted_userids.userid)]

# general_feedback_c1 = general_feedback.text.loc[general_feedback.condition == "c1"]
# general_feedback_c2 = general_feedback.text.loc[general_feedback.condition == "c2"]
# general_feedback_c3 = general_feedback.text.loc[general_feedback.condition == "c3"]

# %%
