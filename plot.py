# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, kstest, mannwhitneyu, shapiro
import statsmodels.api as sm

# %%

result_df = pd.read_csv("dis_results.csv")
# low_nfc = result_df.loc[result_df.nfc < 4.833333, :]
c1 = result_df.loc[result_df.condition == "c1", :]
c2 = result_df.loc[result_df.condition == "c2", :]
c3 = result_df.loc[result_df.condition == "c3", :]

print(c1.describe())
print(c2.describe())
print(c3.describe())

c1_c2 = result_df.loc[result_df.condition.isin(["c1", "c2"]), :]
c2_c3 = result_df.loc[result_df.condition.isin(["c3", "c2"]), :]


# %%

from scipy import stats


df = result_df

for col in ["iuipc", "nfc", "ueq_total", "ueq_pragmatic", "ueq_hedonic"]:
    print(stats.kstest(df[col], stats.norm.cdf))

shapiro(c2.ueq_pragmatic)

# %%


def plot_scatter(alpha=0.3):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    print(axes)

    for i, df, ax in zip(range(3), [c1, c2, c3], axes.flatten()):

        colors = {
            "c1": "red",
            "c2": "green",
            "c3": "blue",
        }

        ax.scatter(
            df.ueq_pragmatic,
            df.ueq_hedonic,
            c=df.condition.map(colors),
            s=df.nfc * 10,
            label=f"c{i + 1}",
            alpha=alpha,
        )

        ax.set_xlabel("ueq pragmatic")
        ax.set_ylabel("ueq hedonic")

        ax.legend()

        plt.tight_layout()

    plt.savefig(
        f"ueq_scatter",
        dpi=800,
        bbox_inches="tight",
    )

    plt.show()


# %%


# %% t-Tests

equal_var = True

i = 1

for df_1, df_2 in zip([c1, c2], [c2, c3]):

    print(f"\n____________________________________________")
    print(f"H{i}")

    print("\nt-Test UEQ Hedonic:")

    print(
        ttest_ind(
            df_1.ueq_hedonic, df_2.ueq_hedonic, equal_var=equal_var, alternative="less"
        )
    )

    print("\nt-Test UEQ Pragmatic:")

    print(
        ttest_ind(
            df_1.ueq_pragmatic,
            df_2.ueq_pragmatic,
            equal_var=equal_var,
            alternative="less",
        )
    )

    print("\nt-Test UEQ Total:")

    print(
        ttest_ind(
            df_1.ueq_total, df_2.ueq_total, equal_var=equal_var, alternative="less"
        )
    )

    i += 1


# %%


# def calc_ols_moderated(df, dependent_dimension, interaction_term):


i = 1
for df in [c1_c2, c2_c3]:
    print(f"\n____________________________________________")
    print(f"H{i}")
    for dim in [
        "ueq_hedonic",
        "ueq_pragmatic",
        "ueq_total",
    ]:
        for interaction_term in ["nfc", "numeracy"]:
            ols_output = sm.OLS.from_formula(
                formula=f"{dim} ~ condition*{interaction_term}",
                data=df,
            ).fit()

            print("\n")
            print(f"Dependent dimension: {dim}")
            print(f"Moderated by: {interaction_term}")
            print(ols_output.summary().tables[1])

    i += 1

# %%

# df = c2_c3


# calc_ols_moderated(df, dependent_dimension="ueq_hedonic", interaction_term="nfc")
# calc_ols_moderated(df, dependent_dimension="ueq_pragmatic", interaction_term="nfc")
# calc_ols_moderated(df, dependent_dimension="ueq_hedonic", interaction_term="numeracy")
# calc_ols_moderated(df, dependent_dimension="ueq_pragmatic", interaction_term="numeracy")
# calc_ols_moderated(df, dependent_dimension="ueq_total", interaction_term="nfc")
# calc_ols_moderated(df, dependent_dimension="ueq_total", interaction_term="numeracy")


# %%

sm.OLS.from_formula(
    formula=f"epsilon_mean ~ condition*iuipc",
    data=df,
).fit().summary().tables[1]


# %%

sm.OLS.from_formula(
    formula=f"epsilon_mean ~ numeracy*iuipc",
    data=c3,
).fit().summary().tables[1]

#%%

sm.OLS.from_formula(
    formula=f"epsilon_mean ~ numeracy*iuipc",
    data=c3,
).fit().summary().tables[1]

# %%
