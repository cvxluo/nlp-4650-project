import re
import pandas as pd


def preprocess_text(text: str):
    """Takes in a tweet and preprocesses the text"""
    url_pattern = r"https?://\S+|www\.\S+"
    user_mention_pattern = r"|@[a-zA-Z0-9_]+"
    hashtag_pattern = r"|#[a-zA-Z0-9_]+"

    # Remove urls, user @ mentions, and hashtags from the text
    # Removing this depends on whether we want the ouput to contain this
    text = re.sub(url_pattern, "", text)
    text = re.sub(user_mention_pattern, "", text)
    text = re.sub(hashtag_pattern, "", text)
    text = text.strip()

    return text


def preprocess(username: str):
    """Takes in a twitter username and preprocesses their data"""
    # Remove retweets
    # Retweets start with RT in the tweet
    df = pd.read_csv(f"./data/{username}.csv")
    is_not_rt = ~df["text"].str.contains("RT")
    df = df[is_not_rt]

    # Preprocess the tweet text
    df["text"] = df["text"].apply(preprocess_text)
    has_len = df["text"].str.len() > 1
    df = df[has_len]
    df = df["text"]

    # Save the cleaned dataframe to a text file
    df.to_csv(f"./data/{username}_cleaned.txt", index=False, header=False)
    return df
