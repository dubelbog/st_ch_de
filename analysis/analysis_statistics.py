import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_bleu = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/blue.csv"
path_added = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/added.tsv"
path_missing = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/missing.tsv"
df = pd.read_csv(path_bleu)
dictionary = []


def creat_plot(title, x_label, y_label, x_axis, y_axis, axis, x_pos, y_pos, color):
    axis[x_pos, y_pos].plot(x_axis, y_axis, 'tab:' + color)
    axis[x_pos, y_pos].set_title(title, fontsize=14)
    axis[x_pos, y_pos].set_xlabel(x_label, fontsize=12)
    axis[x_pos, y_pos].set_ylabel(y_label, fontsize=12)
    axis[x_pos, y_pos].set_xlim(0.0, 1.0)
    if x_label == "BLEU Score" and y_label != 'Number':
        axis[x_pos, y_pos].set_ylim(0.0, 1.0)


def get_extreme_values(x_axis, y_axis, x_name, y_name):
    for x, y in zip(x_axis, y_axis):
        if y == max(y_axis):
            print("max ", x_name, " ", round(y, 3), " bei ", y_name, ": ", x)
        if y == min(y_axis):
            print("min ", x_name, " ", round(y, 3), " bei ", y_name, ": ", x)


def plot_scores_analysis():
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
    df_wer_sum = df.groupby("WER")["BLEU"].mean()
    print(df["BLEU"].mean())

    x_axis = sorted(df.WER.unique())
    y_axis = df.groupby("WER")["BLEU"].count()
    creat_plot('WER Values Verteilung', 'WER Score', 'Number', x_axis, y_axis, axs, 0, 0, "blue")

    y_axis = df.groupby("WER")["BLEU"].mean()
    creat_plot('Mittelwert BLUE Score in Abhängigkeit von WER', 'WER Score', 'Mitelwert BLEU Score', x_axis, y_axis, axs, 1, 0, 'cyan')
    # get_extreme_values(x_axis, y_axis, "BLEU", "WER")

    y_axis = df.groupby("WER")["BLEU"].median()
    creat_plot('Median BLUE Score in Abhängigkeit von WER', 'WER Score', 'Median BLEU Score', x_axis, y_axis, axs, 2, 0, 'grey')
    # get_extreme_values(x_axis, y_axis, "BLEU", "WER")

    y_axis = df.groupby("WER")["BLEU"].max()
    creat_plot('Max BLUE Score in Abhängigkeit von WER', 'WER Score', 'Max BLEU Score', x_axis, y_axis, axs, 3, 0, 'green')
    # get_extreme_values(x_axis, y_axis, "BLEU", "WER")

    y_axis = df.groupby("WER")["BLEU"].min()
    creat_plot('Min BLUE Score in Abhängigkeit von WER', 'WER Score', 'Min BLEU Score', x_axis, y_axis, axs, 4, 0, 'grey')

    x_axis = sorted(df.BLEU.unique())
    y_axis = df.groupby("BLEU")["WER"].count()
    creat_plot('BLUE Values Verteilung', 'BLEU Score', 'Number', x_axis, y_axis, axs, 0, 1, 'blue')

    y_axis = df.groupby("BLEU")["WER"].mean()
    creat_plot('Mittelwert WER Score in Abhängigkeit von BLEU', 'BLEU Score', 'Mitelwert WER Score', x_axis, y_axis, axs, 1, 1, 'cyan')
    y_axis = df.groupby("BLEU")["WER"].median()
    creat_plot('Median WER Score in Abhängigkeit von BLEU', 'BLEU Score', 'Mitelwert WER Score', x_axis, y_axis, axs, 2, 1, 'grey')
    y_axis = df.groupby("BLEU")["WER"].max()
    creat_plot('Max WER Score in Abhängigkeit von BLEU', 'BLEU Score', 'Mitelwert WER Score', x_axis, y_axis, axs, 3, 1, 'green')
    y_axis = df.groupby("BLEU")["WER"].min()
    creat_plot('Min WER Score in Abhängigkeit von BLEU', 'BLEU Score', 'Mitelwert WER Score', x_axis, y_axis, axs, 4, 1, 'grey')
    plt.show()
    fig.savefig('scores_analysis.jpg')


plot_scores_analysis()


def plot_words_occurrences(path, label, title, save_file):
    df_added = pd.read_table(path)
    labels = list(df_added["word"][:30])
    print(labels)
    occurrences = list(df_added["counts"][:30])
    print(occurrences)
    x = np.arange(len(labels))
    # width of the bars
    width = 0.7
    fig, ax = plt.subplots(figsize=(15, 10))
    words = ax.bar(x, occurrences, width, label=label)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Occurrences')
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(ax.get_xticks(), rotation=90, fontsize=14)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(words, padding=5)
    fig.tight_layout()
    plt.show()
    fig.savefig(save_file)


def plot_occurrences():
    plot_words_occurrences(path_added, 'Hinzugefügte Wörter',
                           '30 Wörter, welche am häufigsten in der Hypothese hinzugefügt wurden', 'words_added.jpg')

    plot_words_occurrences(path_missing, 'Fehlende Wörter', '30 Wörter, welche am häufigsten in der Hypothese fehlen',
                           'words_missing.jpg')


# plot_occurrences()


def missing_vs_added():
    com_file_path = "/Users/bdubel/Documents/ZHAW/BA/st_ch_de/resources/scores_analysis/100/comparison.tsv"
    com_file = open(com_file_path, "w")
    df_added = pd.read_table(path_added)
    labels_added = list(df_added["word"])
    df_missing = pd.read_table(path_missing)
    labels_missing = list(df_missing["word"])
    intersection = set(labels_added).intersection(labels_missing)
    comparison = []
    for common in intersection:
        mis = df_missing.loc[df_missing['word'] == common]
        add = df_added.loc[df_added['word'] == common]
        comparison.append([common, int(str(mis).split()[3]), int(str(add).split()[3])])
        com_file.write(common + ";" + str(mis).split()[3] + ";" + str(add).split()[3] + '\n')

    com_file.close()
    sorted_list = sorted(comparison, key=lambda x : x[1], reverse=True)
    first_30 = np.array(sorted_list[:150])
    first_30 = first_30.T
    labels = first_30[0].tolist()
    men_means = list(map(int, first_30[1].tolist()))
    women_means = list(map(int, first_30[2].tolist()))

    width = 0.4  # the width of the bars
    r1 = np.arange(len(labels))
    r2 = [x + width for x in r1]

    fig, ax = plt.subplots(figsize=(50, 10))
    rects1 = ax.bar(r1, men_means, width, label='Fehlend', color='#636261')
    rects2 = ax.bar(r2, women_means, width, label='Hinzugefügt', color='#0aabf0')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Occurrences')
    ax.set_title('Vergleich hinzugefügt vs. fehlend', fontsize=16)
    ax.set_xticks([r + width for r in range(len(labels))])
    ax.set_xticklabels(ax.get_xticks(), rotation=90, fontsize=14)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    fig.savefig("comparison.jpg")


# missing_vs_added()


# print(df["diff_ref_hyp"].value_counts().nlargest(50))
# print(df["diff_hyp_ref"].value_counts().nlargest(50))
# print(df.groupby("hypothese")["hypothese"].apply(lambda ser: ser.str.contains("es").sum()).nlargest(5))

# plt.xlabel("WER Score")
# plt.ylabel("BLUE Score")
# plt.plot(x_axis, y_axis)

#
# df_wer_max = df.groupby("WER")["BLEU"].max()
# print(df_wer_max)
#
# df_wer_count = df.groupby("WER")["BLEU"].median()
# print(df_wer_count)
# df_wer_sum = df.groupby("WER")["BLEU"].mean()
# print(df_wer_sum)
#
# df_blue_count = df.groupby("BLEU")["WER"].median()
# print(df_blue_count)
# df_blue_sum = df.groupby("BLEU")["WER"].mean()
# print(df_blue_sum)

# bins = pd.cut(df["nr_1"], bins=5, labels=("1", "2", "3", "4", "5"))
# # test = df[["rel_hum", "abs_hum"]].groupby(bins).agg(["mean", "median"])
# print(df[["BLEU", "WER"]].groupby(bins).agg(["mean", "median"]))
#
# # print(df.groupby("WER", sort=False)["hypothese"].apply(lambda ser: ser.str.contains("es").sum()).nlargest(5))
# print(df.groupby("WER", sort=False)["hypothese"].apply(lambda ser: ser.str.contains("es").sum()).nlargest(5))
