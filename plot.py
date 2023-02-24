# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_ind,
    kstest,
    mannwhitneyu,
    shapiro,
    levene,
    kruskal,
    pointbiserialr,
)
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
c1_c3 = result_df.loc[result_df.condition.isin(["c3", "c1"]), :]


# %%

from scipy import stats


df = result_df

for col in ["iuipc", "nfc", "ueq_total", "ueq_pragmatic", "ueq_hedonic"]:
    # print(stats.kstest(df[col], stats.norm.cdf))
    print(shapiro(df[col]))

# %%


# def plot_scatter(alpha=0.3):

alpha = 0.4
fig, ax = plt.subplots(figsize=(15, 10))

# print(axes)

for i, df in zip(range(3), [c1, c2, c3]):

    colors = {
        "c1": "red",
        "c2": "green",
        "c3": "blue",
    }

    ax.scatter(
        df.ueq_pragmatic,
        df.ueq_hedonic,
        c=df.condition.map(colors),
        s=df.numeracy * 20,
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

equal_var = False

i = 1

# for df_1, df_2 in [[c1, c2], [c2, c3]]:
for df_1, df_2 in [[c1, c2], [c2, c3], [c1, c3]]:
    print(f"\n____________________________________________")
    print(f"H{i}")
    for dim in ["ueq_hedonic", "ueq_pragmatic", "ueq_total"]:

        print(f"\nTest {dim}:")

        _, p_value = levene(df_1[dim], df_2[dim])

        if p_value <= 0.05:
            equal_var = False
            print("equal_var = False")
        else:
            equal_var = True

        print(
            mannwhitneyu(
                df_1[dim],
                df_2[dim],
                alternative="less",
            )
        )
    i += 1


# %%


def get_latex_table(
    df, condition, dependent_variable, interaction_term=None, title=None, label=None
):
    mapping = {
        "ueq_hedonic": "UEQ hedonic dimension",
        "ueq_pragmatic": "UEQ pragmatic dimension",
        "ueq_total": "UEQ total score",
        "nfc": "NFC Score",
        "C(numeracy)": "Numeracy Score",
    }

    ols_output = sm.OLS.from_formula(
        # formula=f"{dependent_variable} ~ condition*{interaction_term}",
        formula=f"{dependent_variable} ~ condition*nfc*C(numeracy) - nfc:C(numeracy) - condition:nfc:C(numeracy)",
        # formula=f"{dependent_variable} ~ condition*{interaction_term}",
        data=df,
    ).fit()

    print(f"\\begin{{table}}[]")
    print("    \\centering")
    print(
        ols_output.summary()
        .tables[1]
        .as_latex_tabular()
        .replace("condition[T.c2]", "Visual")
        .replace("condition[T.c3]", "Bayesian")
        .replace("C(numeracy)[T.2]", "LowNumeracy")
        .replace("C(numeracy)[T.3]", "HighNumeracy")
        .replace("C(numeracy)[T.4]", "HighestNumeracy")
        .replace(":", "*")
        .replace("nfc", "NFC")
    )
    print(
        f"\\caption{{OLS Regression Results: {title}. Dependent variable: {mapping[dependent_variable]}.}}"
    )
    print(f"\\label{{tab:{dependent_variable}_{condition}}}")

    print(f"\\end{{table}}")


get_latex_table(c1_c2, "visual", "ueq_hedonic", title="Condition Simple vs. Visual")
get_latex_table(c1_c2, "visual", "ueq_pragmatic", title="Condition Simple vs. Visual")
get_latex_table(c1_c2, "visual", "ueq_total", title="Condition Simple vs. Visual")


get_latex_table(c2_c3, "bayesian", "ueq_hedonic", title="Condition Visual vs. Bayesian")
get_latex_table(
    c2_c3, "bayesian", "ueq_pragmatic", title="Condition Visual vs. Bayesian"
)
get_latex_table(c2_c3, "bayesian", "ueq_total", title="Condition Visual vs. Bayesian")

get_latex_table(c1_c3, "bayesian", "ueq_hedonic", title="Condition Simple vs. Bayesian")
get_latex_table(
    c1_c3, "bayesian", "ueq_pragmatic", title="Condition Simple vs. Bayesian"
)
get_latex_table(c1_c3, "bayesian", "ueq_total", title="Condition Simple vs. Bayesian")

# get_latex_table(c1_c2, "visual", "ueq_pragmatic", "nfc")
# get_latex_table(c1_c2, "visual", "ueq_hedonic", "C(numeracy)")
# get_latex_table(c1_c2, "visual", "ueq_pragmatic", "C(numeracy)")
# get_latex_table(c2_c3, "bayesian", "ueq_hedonic", "nfc")
# get_latex_table(c2_c3, "bayesian", "ueq_pragmatic", "nfc")
# get_latex_table(c2_c3, "bayesian", "ueq_hedonic", "C(numeracy)")
# get_latex_table(c2_c3, "bayesian", "ueq_pragmatic", "C(numeracy)")


# %%

# df = c2_c3


# calc_ols_moderated(df, dependent_dimension="ueq_hedonic", interaction_term="nfc")
# calc_ols_moderated(df, dependent_dimension="ueq_pragmatic", interaction_term="nfc")
# calc_ols_moderated(df, dependent_dimension="ueq_hedonic", interaction_term="numeracy")
# calc_ols_moderated(df, dependent_dimension="ueq_pragmatic", interaction_term="numeracy")
# calc_ols_moderated(df, dependent_dimension="ueq_total", interaction_term="nfc")
# calc_ols_moderated(df, dependent_dimension="ueq_total", interaction_term="numeracy")


# %%

for i in range(1, 6):
    print(
        sm.OLS.from_formula(
            formula=f"epsilon_{i} ~ condition*iuipc",
            data=c2_c3,
            # data=c3,
        )
        .fit()
        .summary()
        .tables[1]
    )

# %%

for df in [c1, c2, c3, c1_c2, c2_c3]:
    print(
        sm.OLS.from_formula(
            formula=f"epsilon_mean ~ condition*nfc",
            # formula=f"epsilon_mean ~ iuipc",
            data=df,
        )
        .fit()
        .summary()
        .tables[1]
    )


# %%

for df in [c1, c2, c3, c1_c2, c2_c3]:
    print(
        sm.OLS.from_formula(
            # formula=f"epsilon_mean ~ condition*iuipc",
            # formula=f"epsilon_mean ~ condition*iuipc*C(numeracy) + condition:iuipc + condition:C(numeracy)",
            formula=f"epsilon_mean ~ condition*iuipc*nfc",
            data=df,
        )
        .fit()
        .summary()
        .tables[1]
    )

# %%

sm.OLS.from_formula(
    formula=f"epsilon_mean ~ condition*iuipc",
    data=c2,
    # data=c3,
).fit().summary().tables[1]

# %%

sm.OLS.from_formula(
    formula=f"epsilon_mean ~ condition*iuipc",
    data=c3,
    # data=c3,
).fit().summary().tables[1]


# %%


import numpy as np
from matplotlib.patches import PathPatch


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


# %%


# df = c3
import seaborn as sns


for ueq_score in ["ueq_pragmatic", "ueq_hedonic", "ueq_total"]:

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

    for df, ax in zip((c1, c2, c3), axes):

        lowest_numeracy = df[ueq_score].loc[df.numeracy == 1]
        low_numeracy = df[ueq_score].loc[df.numeracy == 2]
        high_numeracy = df[ueq_score].loc[df.numeracy == 3]
        highest_numeracy = df[ueq_score].loc[df.numeracy == 4]

        ax.boxplot(
            np.array([lowest_numeracy, low_numeracy, high_numeracy, highest_numeracy]),
            widths=0.4,
        )
        ax.set_xlabel("Numeracy")
        # ax.set_ylabel(ueq_score)
        ax.set_ylim([1, 7])

    # plt.title("ueq_hedonic")
    fig.suptitle(ueq_score, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
    plt.savefig(f"{ueq_score}_boxplot.png", facecolor="white", transparent=False)


# %% IUIPC Scatter


alpha = 0.4


x_values = np.linspace(1, 7, 500)

for numeracy in [1, 2, 3, 4]:
    # for nfc_low, nfc_high in zip(range()):

    fig, ax = plt.subplots()

    c1 = result_df.loc[
        (result_df.numeracy == numeracy) & (result_df.condition == "c1"), :
    ]

    c2 = result_df.loc[
        (result_df.numeracy == numeracy) & (result_df.condition == "c2"), :
    ]

    c3 = result_df.loc[
        (result_df.numeracy == numeracy) & (result_df.condition == "c3"), :
    ]

    sm.OLS.from_formula(
        formula=f"epsilon_mean ~ condition*iuipc",
        data=c1,
        # data=c3,
    ).fit().summary().tables[1]

    sm.OLS.from_formula(
        formula=f"epsilon_mean ~ condition*iuipc",
        data=c2,
        # data=c3,
    ).fit().summary().tables[1]

    sm.OLS.from_formula(
        formula=f"epsilon_mean ~ condition*iuipc",
        data=c3,
        # data=c3,
    ).fit().summary().tables[1]

    for i, df in zip(range(3), [c1, c2, c3]):

        colors = {
            "c1": "red",
            "c2": "green",
            "c3": "blue",
        }

        ax.scatter(
            df.iuipc,
            df.epsilon_mean,
            # c=df.condition.map(colors),
            # s=df.numeracy * 20,
            label=f"c{i + 1}",
            alpha=alpha,
        )

        # print(
        #     sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=c1)
        #     .fit()
        #     .summary()
        #     .tables[1]
        # )

        # print(
        #     sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=c2)
        #     .fit()
        #     .summary()
        #     .tables[1]
        # )

        # print(
        #     sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=c3)
        #     .fit()
        #     .summary()
        #     .tables[1]
        # )

        #  ax.scatter(
        #     lower_numeracy.iuipc,
        #     lower_numeracy.epsilon_mean,
        #     c="red",
        #     # s=df.numeracy * 20,
        #     # label=f"c{i + 1}",
        #     alpha=alpha,
        # )

        ax.set_xlabel("IUIPC")
        ax.set_ylabel("PrivacyLevel")
        ax.set_xlim([1, 7])
        # ax.set_xlim([1, 10])

        if i == 0:
            y_values = -10.4208 + 5.9921 * x_values
        elif i == 1:
            y_values = -20.4435 + 7.8140 * x_values
        elif i == 2:
            y_values = 9.0612 + 2.8686 * x_values

        # if i == 0:
        #     y_values = -5.6713 + 5.3199 * x_values
        # elif i == 1:
        #     y_values = 2.2695 + 4.3563 * x_values
        # elif i == 2:
        #     y_values = 9.5880 + 3.2113 * x_values

        ax.plot(x_values, y_values, c=colors[f"c{i + 1}"])

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            f"iuipc_scatter_{numeracy}",
            dpi=800,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

plt.show()


# %%

alpha = 0.4


x_values = np.linspace(1, 7, 500)

for numeracy in [1, 2, 3, 4]:
    # for nfc_low, nfc_high in zip(range()):

    fig, ax = plt.subplots()

    c1 = result_df.loc[
        (result_df.numeracy == numeracy) & (result_df.condition == "c1"), :
    ]

    c2 = result_df.loc[
        (result_df.numeracy == numeracy) & (result_df.condition == "c2"), :
    ]

    c3 = result_df.loc[
        (result_df.numeracy == numeracy) & (result_df.condition == "c3"), :
    ]

    sm.OLS.from_formula(
        formula=f"epsilon_mean ~ condition*iuipc",
        data=c1,
        # data=c3,
    ).fit().summary().tables[1]

    sm.OLS.from_formula(
        formula=f"epsilon_mean ~ condition*iuipc",
        data=c2,
        # data=c3,
    ).fit().summary().tables[1]

    sm.OLS.from_formula(
        formula=f"epsilon_mean ~ condition*iuipc",
        data=c3,
        # data=c3,
    ).fit().summary().tables[1]

    for i, df in zip(range(3), [c1, c2, c3]):

        colors = {
            "c1": "red",
            "c2": "green",
            "c3": "blue",
        }

        ax.scatter(
            df.iuipc,
            df.epsilon_mean,
            # c=df.condition.map(colors),
            # s=df.numeracy * 20,
            label=f"c{i + 1}",
            alpha=alpha,
        )

        # print(
        #     sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=c1)
        #     .fit()
        #     .summary()
        #     .tables[1]
        # )

        # print(
        #     sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=c2)
        #     .fit()
        #     .summary()
        #     .tables[1]
        # )

        # print(
        #     sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=c3)
        #     .fit()
        #     .summary()
        #     .tables[1]
        # )

        #  ax.scatter(
        #     lower_numeracy.iuipc,
        #     lower_numeracy.epsilon_mean,
        #     c="red",
        #     # s=df.numeracy * 20,
        #     # label=f"c{i + 1}",
        #     alpha=alpha,
        # )

        ax.set_xlabel("IUIPC")
        ax.set_ylabel("PrivacyLevel")
        ax.set_xlim([1, 7])
        # ax.set_xlim([1, 10])

        if i == 0:
            y_values = -10.4208 + 5.9921 * x_values
        elif i == 1:
            y_values = -20.4435 + 7.8140 * x_values
        elif i == 2:
            y_values = 9.0612 + 2.8686 * x_values

        # if i == 0:
        #     y_values = -5.6713 + 5.3199 * x_values
        # elif i == 1:
        #     y_values = 2.2695 + 4.3563 * x_values
        # elif i == 2:
        #     y_values = 9.5880 + 3.2113 * x_values

        ax.plot(x_values, y_values, c=colors[f"c{i + 1}"])

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            f"iuipc_scatter_{numeracy}",
            dpi=800,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

plt.show()


# %% privacy_level ~ condition*iuipc


for df, condition in zip([c1, c2, c3], ["Simple", "Visual", "Bayesian"]):
    print(f"\\begin{{table}}[]")
    print("    \\centering")
    print(
        sm.OLS.from_formula(formula=f"epsilon_mean ~ condition*iuipc", data=df)
        .fit()
        .summary()
        .tables[1]
        .as_latex_tabular()
    )
    print(
        f"\\caption{{OLS Regression Results. Dependent variable: PrivacyLevel. Condition: {condition}}}"
    )
    print(f"\\label{{tab:privacy_level_{condition}_iuipc}}")

    print(f"\\end{{table}}")

# %%
