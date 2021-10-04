import os, sys
import tweepy
from dotenv import load_dotenv

import csv

load_dotenv()

API_KEY = os.environ.get("API_KEY")
API_KEY_SECRET = os.environ.get("API_KEY_SECRET")

profile_name = sys.argv[1]

auth = tweepy.AppAuthHandler(API_KEY, API_KEY_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)
tweets = api.user_timeline(screen_name=profile_name, trim_user=True, exclude_replies=False, include_rts=True, count=3200)
last_tweet_id = tweets[0].id


fields = ['created_at', 'id', 'id_str', 'text', 'truncated', 'entities', 'source', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user', 'geo', 'coordinates', 'place', 'contributors', 'is_quote_status', 'retweet_count', 'favorite_count', 'favorited', 'retweeted', 'possibly_sensitive', 'lang', 'quoted_status_id', 'quoted_status', 'quoted_status_id_str', 'extended_entities'] + ['retweeted_status']

with open(profile_name + '.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    while len(tweets) :
        last_tweet_id = tweets[-1].id
        writer.writerows([dict(tweet._json) for tweet in tweets])
        tweets = api.user_timeline(screen_name=profile_name, max_id=last_tweet_id-1, trim_user=True, exclude_replies=False, include_rts=True, count=3200)
        print(len(tweets))