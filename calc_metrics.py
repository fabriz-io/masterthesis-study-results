# %%

# result_df = None

import ast


import pandas as pd
import numpy as np


attentionCheckItem = 'Damit wir wissen, dass Sie die Fragen gelesen haben, wählen sie bitte die Antwortmöglichkeit "Stimme eher zu" aus. Dies ist ein Aufmerksamkeitstests.'

correctAnswerIndexAC = 1

iuipcItemsWithAttentionCheckSorted = [
    "Beim Online-Datenschutz für Verbraucher:innen geht es in erster Linie um das Recht der Verbraucher:innen, Kontrolle und Autonomie über Entscheidungen auszuüben, wie persönliche Informationen gesammelt, verwendet und weitergegeben werden.",
    'Damit wir wissen, dass Sie die Fragen gelesen haben, wählen sie bitte die Antwortmöglichkeit "Stimme eher zu" aus. Dies ist ein Aufmerksamkeitstests.',
    "Die Kontrolle der Verbraucher:innen über ihre persönlichen Informationen ist die Kernaufgabe des Datenschutzes.",
    "Eine gute Online-Datenschutzrichtlinie für Verbraucher:innen sollte  klar formuliert und auffällig platziert sein.",
    "Es ist mir sehr wichtig, dass ich bewusst und sachkundig darüber informiert bin, wie meine persönlichen Informationen verwendet werden.",
    "Es stört mich, so vielen Online-Unternehmen persönliche Informationen zu geben.",
    "Ich bin besorgt, dass Online-Unternehmen zu viele persönliche Informationen über mich sammeln.",
    "Ich glaube, dass die Online-Privatsphäre durch eine Marketingmaßnahme verletzt wird, wenn Kontrolle verloren geht oder ungewollt reduziert wird.",
    "Normalerweise stört es mich, wenn Online-Unternehmen mich nach persönlichen Informationen fragen.",
    "Unternehmen, die Online Informationen einholen, sollten offenlegen, wie die Daten gesammelt, verarbeitet und genutzt werden.",
    "Wenn mich Online-Unternehmen nach persönlichen Informationen fragen, überlege ich manchmal zweimal, bevor ich sie angebe.",
]


nfcItemsWithAttentionCheckSorted = [
    'Damit wir wissen, dass Sie die Fragen gelesen haben, wählen sie bitte die Antwortmöglichkeit "Stimme eher nicht zu" aus. Dies ist ein Aufmerksamkeitstests.',
    "Denken entspricht nicht dem, was ich unter Spaß verstehe.",
    "Die Aufgabe, neue Lösungen für Probleme zu finden, macht mir wirklich Spaß.",
    "Ich trage nicht gerne die Verantwortung für eine Situation, die sehr viel Denken erfordert.",
    "Ich würde komplizierte Probleme einfachen Problemen vorziehen.",
    "Ich würde lieber eine Aufgabe lösen, die Intelligenz erfordert, schwierig und bedeutend ist, als eine Aufgabe, die zwar irgendwie wichtig ist, aber nicht viel Nachdenken erfordert.",
    "Ich würde lieber etwas tun, das wenig Denken erfordert, als etwas, das mit Sicherheit meine Denkfähigkeit herausfordert.",
]

# 0 (R)    "Denken entspricht nicht dem, was ich unter Spaß verstehe.",
#          "Die Aufgabe, neue Lösungen für Probleme zu finden, macht mir wirklich Spaß.",
# 2 (R)    "Ich trage nicht gerne die Verantwortung für eine Situation, die sehr viel Denken erfordert.",
#          "Ich würde komplizierte Probleme einfachen Problemen vorziehen.",
#          "Ich würde lieber eine Aufgabe lösen, die Intelligenz erfordert, schwierig und bedeutend ist, als eine Aufgabe, die zwar irgendwie wichtig ist, aber nicht viel Nachdenken erfordert.",
# 5 (R)    "Ich würde lieber etwas tun, das wenig Denken erfordert, als etwas, das mit Sicherheit meine Denkfähigkeit herausfordert.",


# 03. Thinking is not my idea of fun. (R)
# 11. I really enjoy a task that involves coming up with
# new solutions to problems.
# 02. I like to have the responsibility of handling a situation
# that requires a lot of thinking.
# 01. I would prefer complex to simple problems.
# 04. I would rather do something that requires little
# thought than something that is sure to challenge my
# thinking abilities. (R)
# 15. I would prefer a task that is intellectual, difficult,
# and important to one that is somewhat important
# but does not require much thought.

# %%


prolific_ids_1 = (
    pd.read_csv("dis_1.csv").rename(columns={"Participant id": "prolificid"}).prolificid
)

prolific_ids_2 = (
    pd.read_csv("dis_2.csv").rename(columns={"Participant id": "prolificid"}).prolificid
)

prolific_ids_3 = (
    pd.read_csv("dis_3.csv").rename(columns={"Participant id": "prolificid"}).prolificid
)


print(prolific_ids_1.isin(prolific_ids_2).sum())
print(prolific_ids_1.isin(prolific_ids_3).sum())
print(prolific_ids_2.isin(prolific_ids_1).sum())
print(prolific_ids_2.isin(prolific_ids_3).sum())
print(prolific_ids_3.isin(prolific_ids_1).sum())
print(prolific_ids_3.isin(prolific_ids_2).sum())
# prolific_ids_ = prolific_ids_80.loc[~prolific_ids_80.isin(prolific_ids_70)]

prolific_ids_concat = pd.concat(
    [prolific_ids_1, prolific_ids_2, prolific_ids_3], axis=0
)
# prolific_ids = prolific_ids.loc[:, ["Participant id"]]

prolific_ids = prolific_ids_concat.unique()


# %%

df_dir = "./dataframes"


welcome = pd.read_csv(f"{df_dir}/landed-instructions.log.csv")


# When landing on the welcome screen twice with same prolificid
# drop the second submission (doubled submission)
welcome = welcome.sort_values("datetime")

welcome = welcome.loc[welcome.prolificid.str.len() == 24]
welcome = welcome.drop_duplicates(subset=["prolificid"], keep="first")
# welcome = welcome.drop_duplicates(subset=["userid", "prolificid"])


# %%
iuipc = (
    pd.read_csv("dataframes/final-answers-iuipc.log.csv")
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
)

user_ids_from_iuipc = iuipc.userid

users = welcome.loc[welcome.userid.isin(user_ids_from_iuipc)]
user = users.sort_values("datetime")

users = users.drop_duplicates(subset=["prolificid"])

users = users.loc[:, ["userid", "condition", "prolificid"]]


# %%

wrongtrie = pd.read_csv("dataframes/wrongtrie.log.csv")

# exclude = [
#     "5cb167c332ca24001ae2fcbd",
#     5e4d2880ad8aac000bfa57fe
# ]


# %% Attention checks ausschließen

# %%

# index_df = ids.merge(
#     welcome, left_on=["Participant id"], right_on=["prolificid"], how="left"
# )

# index_df = (
#     index_df.loc[:, ["userid", "prolificid", "condition", "datetime"]].sort_values(
#         "datetime"
#     )
#     # .drop_duplicates(subset=["prolificid"])
# )

# index_df = index_df.drop_duplicates(subset=["userid", "prolificid", "condition"])

# welcome = welcome.loc[:, ["userid", "prolificid", "condition"]]

# prolific_ids = index_df.prolificid
# user_ids = index_df.userid

# %%

# prolific_id_to_user_id = prolific_ids.loc[:, [""]]
# %% Metrics


# %%

# Generating Result DF

iuipc = (
    pd.read_csv("dataframes/final-answers-iuipc.log.csv")
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
)

iuipc = iuipc.loc[
    iuipc.userid.isin(users.userid),
    ["userid", "condition", "answers", "shuffledquestions"],
]
# .rename(columns={"answers": "iuipcAnswers", "shuffledquestions": "iuipcQuestions"})

nfc = (
    pd.read_csv("dataframes/final-answers-nfc.log.csv")
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
)

ueq = (
    pd.read_csv("dataframes/final-answers-ueqs.log.csv")
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
    .rename(columns={"answers": "ueqAnswers"})
    .loc[:, ["userid", "ueqAnswers"]]
)


numeracy = (
    pd.read_csv("dataframes/final-answers-numeracy.log.csv")
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
    .rename(columns={"answer": "numeracyAnswers"})
    .loc[:, ["userid", "numeracyAnswers"]]
)


ueq.ueqAnswers = ueq.ueqAnswers.apply(lambda x: [int(x) for x in ast.literal_eval(x)])
# numeracy.numeracyAnswers = numeracy.numeracyAnswers.apply(
#     lambda x: [int(x) for x in ast.literal_eval(x)]
# )


def get_sorted_answers(df, metricName):
    answer_list = []
    userid_list = []
    condition_list = []

    for userid, condition, questionListString, answerListString in zip(
        df.userid, df.condition, df.shuffledquestions, df.answers
    ):

        questionList = ast.literal_eval(questionListString)
        answerList = ast.literal_eval(answerListString)

        if not isinstance(questionList, list) or not isinstance(answerList, list):
            print("here")
            print(questionListString)
            print(answerListString)

        else:
            # print(len(questionList))

            sortedIndex = np.argsort(questionList)
            # print(sortedIndex)
            answers = np.array(answerList)[sortedIndex]

            answer_list.append(answers.tolist())
            # print(answers)
            userid_list.append(userid)
            condition_list.append(condition)

    return pd.DataFrame(
        {
            "userid": userid_list,
            "condition": condition_list,
            f"{metricName}AnswersSorted": answer_list,
        }
    )


iuipc_df = get_sorted_answers(iuipc, "iuipc")
nfc_df = get_sorted_answers(nfc, "nfc")
# ueq = get_sorted_answers(ueq, "ueq")


result_df = (
    iuipc_df.merge(nfc_df, on=["userid", "condition"]).merge(ueq).merge(numeracy)
)

# result_df = result_df.loc[result_df.userid.isin(user.userid)]


print(result_df.condition.value_counts())


def calc_iuipc(answer_list):
    if len(answer_list) != 11:
        raise ValueError

    else:
        # print(type(answer_list))
        answer_list.pop(1)
        return np.mean([int(x) for x in answer_list])
    # return 1


def calc_nfc(answer_list):
    if len(answer_list) != 7:
        raise ValueError

    else:
        # print(type(answer_list))
        answer_list.pop(0)
        answer_list = [int(x) for x in answer_list]

        nfc_score = []
        for index, a in enumerate(answer_list):
            if index in [0, 2, 5]:
                nfc_score.append(8 - a)
            else:
                nfc_score.append(a)

    return np.mean(nfc_score)


# %%
import pandas as pd


def calc_numeracy(answer_list):
    """
    ['25', '30', '20', '50']
    """
    answers = ast.literal_eval(answer_list)

    if answers[2]:
        if answers[2] == "20":
            return 4

    if answers[3]:
        if answers[3] != "50":
            return 3
        elif answers[3] == "50":
            return 4

    if answers[1]:
        if answers[1] != "30":
            return 1
        elif answers[1] == "30":
            return 2

    raise ValueError


result_df["iuipc"] = result_df.iuipcAnswersSorted.apply(lambda x: calc_iuipc(x))
result_df["nfc"] = result_df.nfcAnswersSorted.apply(lambda x: calc_nfc(x))
result_df["numeracy"] = result_df.numeracyAnswers.apply(lambda x: calc_numeracy(x))


# %%

result_df.to_csv("dis_results.csv")

# %%
