
def add_labels(df):
    """
    Add label column in the original dataframe
    """
    res = df.copy()
    sentiment_to_label = {"positive": 2, "neutral": 1, "negative": 0}
    res["labels"] = res[0].apply(lambda sentiment: sentiment_to_label[sentiment])
    return res


def add_processed_col(df):
    """
    Add processed column for BERT usage
    """
    res = df.copy()
    list_input = []
    for i in range(len(df[0])):
        string = ""
        string += df[4].iloc[i]
        string += " [SEP] "
        string += df[2].iloc[i]
        string += " [SEP] "
        string += df[1].iloc[i]
        list_input.append(string)
    res["processed_input"] = list_input
    return res

