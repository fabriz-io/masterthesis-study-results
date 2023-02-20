# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

result_df = pd.read_csv("dis_results.csv")
accepted_userids = pd.read_csv("accepted_userids.csv")

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

german_stop_words = stopwords.words("german")

# %% Time Taken for the survey

fig, ax = plt.subplots()

landed_c1 = pd.read_csv("dataframes/landed-c1.log.csv").loc[
    :, ["userid", "datetime", "condition"]
]
landed_c2 = pd.read_csv("dataframes/landed-c2.log.csv").loc[
    :, ["userid", "datetime", "condition"]
]
landed_c3 = pd.read_csv("dataframes/landed-c3.log.csv").loc[
    :, ["userid", "datetime", "condition"]
]

landed_condition = pd.concat([landed_c1, landed_c2, landed_c3]).rename(
    columns={"datetime": "condition_start"}
)

landed_condition = (
    landed_condition.loc[landed_condition.userid.isin(accepted_userids.userid)]
    .sort_values(by="condition_start")
    .drop_duplicates(keep="first", subset=["userid"])
)

landed_thanks = (
    pd.read_csv("dataframes/landed-thanks-svg.log.csv")
    .loc[:, ["userid", "datetime"]]
    .rename(columns={"datetime": "condition_end"})
    .sort_values(by="condition_end")
    .drop_duplicates(keep="first", subset=["userid"])
)

time_condition = pd.merge(landed_condition, landed_thanks, how="left", on=["userid"])


time_condition["elapsed_time"] = (
    time_condition.condition_end - time_condition.condition_start
) / 1000

condition_label_mapping = {
    "c1": "Simple",
    "c2": "Visual",
    "c3": "Bayesian",
}

ax = sns.boxplot(
    time_condition,
    x="condition",
    y="elapsed_time",
    width=0.3,
    order=["c1", "c2", "c3"]
    # labels=time_condition.condition.apply(lambda x: condition_label_mapping[x]),
)

ax.set_xlabel("Condition")
ax.set_xticklabels(["Simple", "Visual", "Bayesian"])
ax.set_ylabel("Time taken in seconds")
# ax.set_yticks()
ax.set_title("Time taken for each Condition")
# ax.legend()
plt.tight_layout()

plt.savefig("time_elapsed.png", dpi=500)

# %% Qualitative Word Cloud Personal Location

personal_location = pd.read_csv("dataframes/freetext-s1.log.csv").loc[
    :, ["userid", "datetime", "freetext"]
]

personal_location = (
    personal_location.loc[personal_location.userid.isin(accepted_userids.userid)]
    .sort_values(by="datetime")
    .drop_duplicates(keep="last", subset=["userid"])
)

personal_location = personal_location.freetext.str.cat(sep=" ")

import string

personal_location = personal_location.translate(
    str.maketrans("", "", string.punctuation)
)

personal_location = personal_location.replace("  ", " ")

personal_location = personal_location.split(" ")

personal_location = [
    item for item in personal_location if item not in german_stop_words
]

personal_location = " ".join(personal_location)

# %%


# for word in german_stop_words:
#     personal_location.freetext = personal_location.freetext.str.replace(
#         f"\s{word}\s", "", regex=True
#     )

# personal_location.freetext.apply(
#     lambda x: [item for item in x if item not in german_stop_words]
# )

wordcloud = WordCloud(
    width=1000,
    height=500,
).generate(personal_location)

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
wordcloud.to_file("wordcloud_personal_place.png")
plt.show()


# %% NFC Score


def violinplot(
    ax,
    column,
    column_name_clean=None,
    title=None,
):

    ax.violinplot(
        [
            result_df[column].loc[result_df.condition == "c1"],
            result_df[column].loc[result_df.condition == "c2"],
            result_df[column].loc[result_df.condition == "c3"],
        ],
        showmedians=True,
    )

    ax.set_title(f"{column_name_clean} Score in each condition")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Simple", "Visual", "Bayesian"])
    ax.set_ylabel(column_name_clean)


def plot_numeracy(ax):
    bar_width = 0.4
    ax = sns.histplot(
        data=result_df,
        x="numeracy",
        # y="condition",
        hue="condition",
        multiple="dodge",
        # kind="hist",
        # shrink=0.8,
        binwidth=bar_width,
        # discrete=True,
        # common_bins=True,
        # legend=False,
    )

    ax.set_xticks(np.array([1, 2, 3, 4]) + (bar_width / 2))
    ax.set_xticklabels(
        ["LowestNumeracy", "LowNumeracy", "HighNumeracy", "HighestNumeracy"],
        rotation=90,
    )
    ax.set_xlim(0, 5)
    ax.set_title("Numeracy Score in each condition")


# %% Descriptives

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))


configs = [
    ["nfc", "NFC"],
    ["iuipc", "IUIPC"],
    ["ueq_pragmatic", "UEQ Pragmatic Dimension"],
    ["ueq_hedonic", "UEQ Hedonic Dimension"],
    ["ueq_hedonic", "UEQ Hedonic Dimension"],
]

for i, ax in enumerate(axes.flatten()[:-1]):
    violinplot(ax, configs[i][0], configs[i][1])

plot_numeracy(axes.flatten()[-1])

fig.suptitle("Descriptives for all measures", fontsize=16)
plt.tight_layout(pad=0.5)
plt.subplots_adjust(left=0.1, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)
plt.savefig("descriptives_measures.png")
plt.show()


# %% Plot privacy levels for different locations

fig, ax = plt.subplots()

privacy_level_c1 = pd.read_csv("dataframes/finalstate-c1.log.csv").loc[
    :, ["userid", "datetime", "component", "finalstate"]
]
privacy_level_c2 = pd.read_csv("dataframes/finalstate-c2.log.csv").loc[
    :, ["userid", "datetime", "component", "finalstate"]
]
privacy_level_c3 = pd.read_csv("dataframes/finalstate-c3.log.csv").loc[
    :, ["userid", "datetime", "component", "finalstate"]
]

privacy_level = (
    pd.concat([privacy_level_c1, privacy_level_c2, privacy_level_c3])
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
)

privacy_level = privacy_level.loc[privacy_level.userid.isin(accepted_userids.userid)]

import ast

privacy_level["home"] = privacy_level.finalstate.apply(lambda x: ast.literal_eval(x)[0])
privacy_level["cafe"] = privacy_level.finalstate.apply(lambda x: ast.literal_eval(x)[1])
privacy_level["private"] = privacy_level.finalstate.apply(
    lambda x: ast.literal_eval(x)[2]
)
privacy_level["workplace"] = privacy_level.finalstate.apply(
    lambda x: ast.literal_eval(x)[3]
)
privacy_level["metro"] = privacy_level.finalstate.apply(
    lambda x: ast.literal_eval(x)[4]
)


privacy_level_melted = privacy_level.loc[
    :, ["component", "home", "cafe", "private", "workplace", "metro"]
].melt(id_vars="component", var_name="location")

ax = sns.boxplot(
    data=privacy_level_melted,
    x="location",
    y="value",
    hue="component",
    linewidth=0.5,
    hue_order=["c1", "c2", "c3"],
)

fig.suptitle("Chosen Privacy Levels in each condition")

ax.set_ylabel("Privacy Level")
ax.legend(loc="upper right")

plt.savefig("privacy_levels.png", dpi=500)

plt.show()

# %%
