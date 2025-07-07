
import os
import re
import tweepy
import pandas as pd
from dateutil import parser


def _initAPI():
    consumer_info = (os.environ["CONSUMER_KEY"], os.environ["CONSUMER_SECRET"])
    access_info = (os.environ["ACCESS_TOKEN"], os.environ["ACCESS_SECRET"])
    auth = tweepy.OAuthHandler(**consumer_info)
    auth.set_access_token(**access_info)
    return auth

def format_tweets(alltweets):
    """ format tweets to a list, keeping retweets """
    handle = alltweets[0]['user']['screen_name'] # twitter handle of current user
    # format created date as timestamp
    to_time = lambda x: parser.parse(x).strftime('%Y-%m-%d %H:%M:%S')

    lst = []
    for tweet in alltweets:
        found_tweet = tweet.get("retweeted_status", "")
        if found_tweet:
            tweet_data = [[
                found_tweet['id'],
                to_time(found_tweet['created_at']),
                handle,
                found_tweet['full_text']
            ]]
            lst.extend(tweet_data)

    return lst

def get_all_tweets(user_names):
    """ scrape the most recent tweets for given accounts. If it's a retweet, grab full text 
    from the original tweet. Return all tweets as a df """
    # initialize connection with json parser api to automatically output json dicts
    auth = _initAPI()
    api = tweepy.API(auth,
                    parser=tweepy.parsers.JSONParser(),
                    wait_on_rate_limit=True)

    # holds all tweets of all users
    all_data = []
    for user_name in user_names:
        # holds all most recent tweets of current user
        alltweets = []
        # get first 200 extended tweets from current user
        new_tweets = api.user_timeline(screen_name=user_name, count=200, tweet_mode="extended")
        alltweets.extend(new_tweets)

        # store tweet id of the last tweet pulled -1 to set new data scraping starting point
        oldest = alltweets[-1]['id'] - 1

        handle = alltweets[0]['user']['screen_name'] # twitter handle of current user
        print(f"Grabbing tweets from {handle}...")

        #keep grabbing tweets until there are none left (limit of 3250/user reached)
        while len(new_tweets) > 0:
            # pull next 200 tweets up to the tweet id of the last tweet of previous batch
            new_tweets = api.user_timeline(screen_name = user_name, count=200, tweet_mode ="extended", max_id=oldest)

            #save most recent tweets
            alltweets.extend(new_tweets)
            #update the max. tweet id for new starting point
            oldest = alltweets[-1]['id'] - 1

            if len(alltweets)%500 == 0:
                print(f"...{len(alltweets)} tweets pulled so far")

        print(f"{len(alltweets)} total tweets pulled from {handle}\n")

        data = format_tweets(alltweets)
        all_data.extend(data)

    df = pd.DataFrame(all_data, columns = ['tweet id', 'created at', 'screen name', 'full text'])
    return df

def clean_tweet(x):
    """ clean the tweets to remove accounts and links """ 

    stop_words = ['says','like','said','tweet','tweeted']
    # remove non-alphanumeric chars, punctuation, hashtags/mentions
    regex = "(@([A-Za-z0-9._-]+))|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([()!?])|(\[.*?\])|(#[A-Za-z0-9_]+)"    
    clean=' '.join(re.sub(regex," ",x).split())
    clean = re.sub(r'\d+', '', clean)                               # remove digits
    clean = [w for w in clean if w not in stop_words]               # only keep non-stop words
    clean = ''.join(word for word in clean)
    return clean