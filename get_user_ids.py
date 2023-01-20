# %%


import pandas as pd

# %%
df = pd.read_csv(
    "/Users/rizio/repos/masterthesis/study_results/dataframes/landed-numeracy.log.csv"
)

# %%


df.condition.value_counts()  # %%

# %%
