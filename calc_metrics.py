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


ueq_pragmatic = ["behindernd", "kompliziert", "ineffizient", "verwirrend"]
ueq_hedonic = ["langweilig", "uninteressant", "konventionell", "herkömmlich"]

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
    pd.read_csv("dis_60.csv")
    .rename(columns={"Participant id": "prolificid"})
    .prolificid
)

prolific_ids_2 = (
    pd.read_csv("dis_70.csv")
    .rename(columns={"Participant id": "prolificid"})
    .prolificid
)

prolific_ids_3 = (
    pd.read_csv("dis_98.csv")
    .rename(columns={"Participant id": "prolificid"})
    .prolificid
)

prolific_ids_4 = (
    pd.read_csv("dis_100.csv")
    .rename(columns={"Participant id": "prolificid"})
    .prolificid
)

print(prolific_ids_1.isin(prolific_ids_3).sum())
print(prolific_ids_2.isin(prolific_ids_3).sum())


participated_double = pd.concat(
    [
        prolific_ids_1.loc[prolific_ids_1.isin(prolific_ids_3)],
        prolific_ids_2.loc[prolific_ids_2.isin(prolific_ids_3)],
    ]
)


print(prolific_ids_1.isin(prolific_ids_2).sum())
print(prolific_ids_1.isin(prolific_ids_4).sum())
print(prolific_ids_2.isin(prolific_ids_1).sum())
print(prolific_ids_2.isin(prolific_ids_4).sum())
print(prolific_ids_3.isin(prolific_ids_1).sum())
print(prolific_ids_3.isin(prolific_ids_2).sum())
print(prolific_ids_3.isin(prolific_ids_4).sum())
# prolific_ids_ = prolific_ids_80.loc[~prolific_ids_80.isin(prolific_ids_70)]

prolific_ids_concat = pd.concat(
    [prolific_ids_1, prolific_ids_2, prolific_ids_3, prolific_ids_4], axis=0
)
# prolific_ids = prolific_ids.loc[:, ["Participant id"]]

prolific_ids = prolific_ids_concat.unique()


# %%

df_dir = "./dataframes"


# welcome = pd.read_csv(f"{df_dir}/landed-instructions.log.csv")


# # When landing on the welcome screen twice with same prolificid
# # drop the second submission (doubled submission)
# welcome = welcome.sort_values("datetime")

# welcome = welcome.loc[welcome.prolificid.str.len() == 24]
# welcome = welcome.drop_duplicates(subset=["prolificid"], keep="first")
# # welcome = welcome.drop_duplicates(subset=["userid", "prolificid"])


# # %%
# iuipc = (
#     pd.read_csv("dataframes/final-answers-iuipc.log.csv")
#     .sort_values("datetime")
#     .drop_duplicates(subset=["userid"], keep="last")
# )

# user_ids_from_iuipc = iuipc.userid

# users = welcome.loc[welcome.userid.isin(user_ids_from_iuipc)]
# user = users.sort_values("datetime")

# users = users.drop_duplicates(subset=["prolificid"])

# users = users.loc[:, ["userid", "condition", "prolificid"]]


# %%

wrongtrie = (
    pd.read_csv("dataframes/wrongtrie.log.csv").groupby("userid").sum().reset_index()
)


ac_failed = pd.read_csv("dataframes/warning-ac-failed.log.csv")

# exclude = [
#     "2905f5f2-5b75-49dc-adf5-64990ebdbb71",
#     "52bd1719-f363-4f9c-98b6-c5c9b07e8b95",
#     "5c4a1012-75ef-4c31-ab92-9d179e6cbf7b",
#     "ab03fc57-72c3-4efd-ac5e-219265800789",
#     "ae783353-5fb9-4e5e-94e1-2ef93f5fd8d0",
#     "f89148b8-4bd0-41fc-9092-b8cf591841a9",
#     "fe8c3545-0b3a-4489-919e-fe41edc4afdb",
#     "c57accc4-2d94-4545-9167-baea3f50b58e",  # ac failed twice
# ]

exclude = pd.concat(
    [
        wrongtrie.userid.loc[wrongtrie.noOfTrie >= 2],
        ac_failed.userid.loc[ac_failed.prolificid.str.len() == 24],
    ]
)

# (wrongtrie.groupby("userid").sum().noOfTrie >= 2)

# %% Attention checks ausschließen


s1 = pd.read_csv("dataframes/freetext-s1.log.csv")


s1 = s1.loc[s1.prolificid.isin(prolific_ids)]

s1 = s1.sort_values(by="datetime")

s1 = s1.drop_duplicates(subset=["userid"], keep="last")

accept_duplicated = s1.loc[s1.prolificid.duplicated()]
accept_duplicated["date"] = pd.to_datetime(s1["datetime"])

# %%

s1 = s1.drop_duplicates(subset=["prolificid"], keep="first")

excluded_users = s1.loc[s1.userid.isin(exclude)]

s1 = s1.loc[~s1.userid.isin(exclude)]

# s1 = s1.loc[~s1.userid.isin(exclude)]

users = s1

# %%a

# ac_nfc = nfc_df.nfcAnswersSorted.apply(lambda x: int(x[0])).value_counts()
# ac_iuipc = iuipc_df.iuipcAnswersSorted.apply(lambda x: int(x[1])).value_counts()

# user_id_failed_iuipc = iuipc_df.userid.loc[
#     iuipc_df.iuipcAnswersSorted.apply(lambda x: int(x[1])) != 5
# ]

# user_id_failed_nfc = nfc_df.userid.loc[
#     nfc_df.nfcAnswersSorted.apply(lambda x: int(x[0])) != 3
# ]


# %%


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

# iuipc = iuipc.loc[
#     iuipc.userid.isin(users.userid),
#     ["userid", "condition", "answers", "shuffledquestions"],
# ]
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
    # .rename(columns={"answers": "ueqAnswers", "shuffledquestions": "ueqQuestions"})
    # .loc[:, ["userid", "ueqAnswers", "ueqQuestions"]]
)


numeracy = (
    pd.read_csv("dataframes/final-answers-numeracy.log.csv")
    .sort_values("datetime")
    .drop_duplicates(subset=["userid"], keep="last")
    .rename(columns={"answer": "numeracyAnswers"})
    .loc[:, ["userid", "numeracyAnswers"]]
)


# ueq.shuffledquestions = ueq.shuffledquestions.apply(
#     lambda x: [x[0] for x in ast.literal_eval(x)]
# )


# %%


def get_sorted_answers(df, metricName):
    question_list = []
    answer_list = []
    userid_list = []
    condition_list = []

    for userid, condition, questionListString, answerListString in zip(
        df.userid, df.condition, df.shuffledquestions, df.answers
    ):

        if metricName == "ueq":
            questionList = [x[0] for x in ast.literal_eval(questionListString)]
        else:
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
            questions = np.array(questionList)[sortedIndex]

            answer_list.append(answers.tolist())
            question_list.append(questions.tolist())
            userid_list.append(userid)
            condition_list.append(condition)

    return pd.DataFrame(
        {
            "userid": userid_list,
            "condition": condition_list,
            f"{metricName}AnswersSorted": answer_list,
            f"{metricName}QuestionsSorted": question_list,
        }
    )


iuipc_df = get_sorted_answers(iuipc, "iuipc")
nfc_df = get_sorted_answers(nfc, "nfc")
ueq = get_sorted_answers(ueq, "ueq")
ueq.ueqAnswersSorted = ueq.ueqAnswersSorted.apply(lambda x: [int(a) for a in x])
# (nfc_df.nfcAnswersSorted.apply(lambda x: int(x.pop(0)))).value_counts()

# %%

result_df = (
    iuipc_df.merge(nfc_df, on=["userid", "condition"]).merge(ueq).merge(numeracy)
)

# result_df = result_df.loc[result_df.userid.isin(user.userid)]

#%%


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


def calc_ueq_pragmatic(answer_list):
    # print(answer_list)
    boolean_indexer_sorted = [True, False, True, True, False, False, False, True]

    pragmatic_score = np.mean(np.array(answer_list)[boolean_indexer_sorted])

    return pragmatic_score


def calc_ueq_hedonic(answer_list):
    # print(answer_list)
    boolean_indexer_sorted = [False, True, False, False, True, True, True, False]

    hedonic_score = np.mean(np.array(answer_list)[boolean_indexer_sorted])

    return hedonic_score


def calc_ueq_total(answer_list):
    # print(answer_list)
    ueq_score = np.mean(np.array(answer_list))
    return ueq_score


# [False, True, False, False, True, True, True, False]


result_df["iuipc"] = result_df.iuipcAnswersSorted.apply(lambda x: calc_iuipc(x))
result_df["nfc"] = result_df.nfcAnswersSorted.apply(lambda x: calc_nfc(x))
result_df["numeracy"] = result_df.numeracyAnswers.apply(lambda x: calc_numeracy(x))
result_df["ueq_pragmatic"] = result_df.ueqAnswersSorted.apply(
    lambda x: calc_ueq_pragmatic(x)
)
result_df["ueq_hedonic"] = result_df.ueqAnswersSorted.apply(
    lambda x: calc_ueq_hedonic(x)
)
result_df["ueq_total"] = result_df.ueqAnswersSorted.apply(lambda x: calc_ueq_total(x))


# %%

import pandas as pd

# epsilon_c1 = pd.read_csv("dataframes/finalstate-c1.log.csv")
# epsilon_c2 = pd.read_csv("dataframes/finalstate-c2.log.csv")
# epsilon_c3 = pd.read_csv("dataframes/finalstate-c3.log.csv")

epsilon = pd.concat(
    [
        pd.read_csv("dataframes/finalstate-c1.log.csv"),
        pd.read_csv("dataframes/finalstate-c2.log.csv"),
        pd.read_csv("dataframes/finalstate-c3.log.csv"),
    ]
)

# %%

epsilon["epsilon_mean"] = epsilon.finalstate.apply(
    lambda x: np.mean(ast.literal_eval(x))
)
epsilon["epsilon_1"] = epsilon.finalstate.apply(lambda x: ast.literal_eval(x)[0])
epsilon["epsilon_2"] = epsilon.finalstate.apply(lambda x: ast.literal_eval(x)[1])
epsilon["epsilon_3"] = epsilon.finalstate.apply(lambda x: ast.literal_eval(x)[2])
epsilon["epsilon_4"] = epsilon.finalstate.apply(lambda x: ast.literal_eval(x)[3])
epsilon["epsilon_5"] = epsilon.finalstate.apply(lambda x: ast.literal_eval(x)[4])


epsilon = epsilon.sort_values("datetime").drop_duplicates("userid", keep="last")

result_df = result_df.merge(
    epsilon.loc[
        :,
        [
            "userid",
            "epsilon_mean",
            "epsilon_1",
            "epsilon_2",
            "epsilon_3",
            "epsilon_4",
            "epsilon_5",
        ],
    ]
)

# %%

result_df = result_df.loc[result_df.userid.isin(s1.userid)]
print(result_df.condition.value_counts())
result_df.to_csv("dis_results.csv")

# Save accepted userids

result_df.userid.to_csv("accepted_userids.csv", index=False)

# %%

dis_60 = pd.read_csv("dis_60.csv")
dis_70 = pd.read_csv("dis_70.csv")
dis_98 = pd.read_csv("dis_98.csv")
dis_100 = pd.read_csv("dis_100.csv")
dis_60["bulk"] = "60"
dis_70["bulk"] = "70"
dis_98["bulk"] = "98"
dis_100["bulk"] = "100"

dis_submissions = pd.concat([dis_60, dis_70, dis_98, dis_100])

dis_submissions = dis_submissions.sort_values(by="Completed at")

duplicated_submissions = dis_submissions.loc[
    dis_submissions["Participant id"].duplicated(keep=False)
]

duplicated_submissions.sort_values(by="Completed at")

# %%
reject = duplicated_submissions.loc[
    ~duplicated_submissions["Participant id"].isin(accept_duplicated.prolificid)
]

reject = reject.loc[reject["Participant id"].duplicated(keep="first")]


exclude = dis_submissions.loc[
    dis_submissions["Participant id"].isin(excluded_users.prolificid)
]


reject_ids = pd.concat([reject["Participant id"], exclude["Participant id"]])


accept_dis_98 = dis_98.loc[~dis_98["Participant id"].isin(reject_ids)]

# %%
