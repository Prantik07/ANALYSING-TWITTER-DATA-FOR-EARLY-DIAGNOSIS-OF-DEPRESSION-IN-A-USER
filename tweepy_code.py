# import tweepy
import tweepy as tw
import re    # for regular expressions 
import nltk  # for text manipulation 
import string # for text manipulation 
import warnings 
import pickle
from nltk.stem.porter import *
import statistics


with open('archive/NaiveBayes_79.pkl', 'rb') as file:
    data = pickle.load(file)
loaded_model = data['model']
loaded_cv = data['cv']


'''NLP Section'''

stopwords=nltk.corpus.stopwords.words('english') 
stemmer = PorterStemmer() 

# Function to process the tweet
def process_tweet(tweet):
    
    processed_tweet = tweet.replace("@", "") 
    processed_tweet = re.sub(r'http\S+', '', processed_tweet)
    processed_tweet = processed_tweet.replace("[^a-zA-Z]", " ") 
    processed_tweet = ' '.join([word for word in processed_tweet.split() if word not in stopwords])
    processed_tweet = processed_tweet.split()
    processed_tweet = [stemmer.stem(word) for word in processed_tweet]
    processed_tweet = [word for word in processed_tweet if len(word) > 3]
    processed_tweet = ' '.join(processed_tweet)
    
    return processed_tweet


''' Twitter API Section '''

# your Twitter API key and API secret
TWITTER_CONSUMER_KEY = "LHatrkPNSc4bj3hr5zqE1XRda"
TWITTER_CONSUMER_SECRET = "L9bmSBk79VQFCx8K9YaxuXQb89UO8qCHOlf7c9kPWNvi7GOCtE"
TWITTER_ACCESS_TOKEN = '1511559074889953280-6xmAtS4SgzseFCAN6eLzDOI61fVsUJ'
TWITTER_ACCESS_TOKEN_SECRET = 'tvpWKeK5W8hUIMiSXkbikQfx0GMN3Hh09LVOQQssgULUZ'

#authenticate
auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True)


tweets_list = tweepy.Cursor(api.search_tweets, q="from:@elonmusk", 
                            tweet_mode='extended', lang='en').items(10)

'''
df = pd.DataFrame(columns=['tweet'])
for tweet in tweets_list:
    text = tweet._json["full_text"]
    text = str(text)

    df.loc[len(df.index)] = [text] '''
    
preds = []    
for tweet in tweets_list:
    text = tweet._json["full_text"]
    print(text)
    text = process_tweet(text)
    X = loaded_cv.transform([text])
    pred = loaded_model.predict(X)
    preds.append(pred[0])
    
max_ele = statistics.mode(preds)
if max_ele == 0:
    print("negative")
elif max_ele == 1:
    print('neutral')
else: print("positive")
    
    
    
    
    

    
