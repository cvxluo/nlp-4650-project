import re

import pandas as pd


def preprocess(text):
    url_pattern = r"https?://\S+|www\.\S+"
    link_pattern = r"\w+.(com|org|gov)\n.*\n.*"
    video_metadata_pattern = r"(\d+(\.\d+)?(K|M)?) views.*"
    video_duration_pattern = r"\d:\d+"
    metadata_pattern = r"\n(\d+(\.\d+)?(K|M)?)\n(\d+(\.\d+)?(K|M)?)(\n(\d+(.\d+)?(K|M)?))?"
    only_metdata_pattern = r"(\d+(\.\d+)?(K|M)?)\n(\d+(\.\d+)?(K|M)?)\n(\d+(\.\d+)?(K|M)?)"
    thread_pattern = r"Show this thread"
    embedded_tweet_pattern = r"\n(.*)\n@\w*\n · .*"
    poll_pattern = r"\n.*\n(\d+(\.\d+)?)%.*"
    poll_result_pattern = r"\d{1,3}(,\d{3})*(\.\d+)? votes\n·\nFinal results"

    text = re.sub(embedded_tweet_pattern, "", text)
    text = re.sub(url_pattern, "", text)
    text = re.sub(metadata_pattern, "", text)
    text = re.sub(only_metdata_pattern, "", text)
    text = re.sub(video_metadata_pattern, "", text)
    text = re.sub(video_duration_pattern, "", text)
    text = re.sub(thread_pattern, "", text)
    text = re.sub(poll_pattern, "", text)
    text = re.sub(poll_result_pattern, "", text)
    text = re.sub(link_pattern, "", text)
    text = re.sub("/", "", text)
    text = text.strip()

    return text


def preprocess_dataset(df: pd.DataFrame):
    is_not_rt = ~df["Embedded_text"].str.contains("RT")
    is_not_reply = ~df["Embedded_text"].str.contains("Replying to")
    is_not_unavailable = ~df["Emebdded_text"].str.contains("This Tweet is unavailable")

    pipeline = [
        is_not_rt,
        is_not_reply,
        is_not_unavailable
    ]

    for filter in pipeline:
        df = df[filter]

    df['text'] = df['Embedded_text'].apply(preprocess)
    return df[['text']]

def load_dataset(username: str, split=[.8, .1, .1], data_dir='data'):
    assert round(sum(split), 2) == 1 and len(split) >= 2
    train_df, test_df, val_df = None, None, None
    df = pd.read_csv(f'{data_dir}/{username}.csv')
    df = preprocess_dataset(df)
    train_split = split[0]
    val_split = split[1]
    train_end = int(train_split * len(df.index))
    validate_end = int(val_split * len(df.index)) + train_end
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:validate_end], df.iloc[validate_end:]
    return train_df, val_df, test_df
