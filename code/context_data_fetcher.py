import requests
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.preprocessing import MinMaxScaler
import praw
from prawcore.exceptions import Redirect
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Accessing variables
GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent="covid_analysis_app"
)

# Initialize the language model and tokenizer (COVID-specific BERT model)
tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
model = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def fetch_contextual_data(api_key, query, region, source='GNews', start_date=None, end_date=None):
    """Fetch historical news or tweets about COVID-19 for a given region and date range."""
    if source == 'GNews':
        url = f'https://gnews.io/api/v4/search?q={query}&lang=en&country={region}&token={api_key}'
        if start_date and end_date:
            url += f"&from={start_date}&to={end_date}"
        headers = {}
    elif source == 'Twitter':
        url = f'https://api.twitter.com/2/tweets/search/recent?query={query}%20place_country:{region}'
        if start_date:
            url += f"&start_time={start_date}T00:00:00Z"  # start_time in ISO format
        if end_date:
            url += f"&end_time={end_date}T00:00:00Z"  # end_time in ISO format
        headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None

def fetch_reddit_data(subreddit_name, post_type="top", limit=30):
    """Fetch top or hot posts from a given subreddit."""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        if post_type == "top":
            posts = subreddit.top(limit=limit)
        elif post_type == "hot":
            posts = subreddit.hot(limit=limit)
        else:
            posts = subreddit.new(limit=limit)

        # Extract the needed fields from each post
        return [
            {
                "title": post.title,
                "body": post.selftext,
                "upvotes": post.score,
                "num_comments": post.num_comments
            }
            for post in posts
        ]
    except Redirect:
        print(f"Warning: Subreddit '{subreddit_name}' does not exist or is restricted.")
        return []  # Return an empty list if subreddit is unavailable

def process_text_data(text_data, max_length=510):
    """Generate embeddings using a pre-trained language model, handling long inputs."""
    embeddings = []
    for text in text_data:
        # Tokenize and handle long text by chunking
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        num_tokens = tokens.input_ids.shape[1]

        if num_tokens > max_length:
            chunk_embeddings = []
            for i in range(0, num_tokens, max_length):
                chunk = tokenizer.decode(tokens.input_ids[0, i:i+max_length])
                chunk_tokens = tokenizer(chunk, return_tensors="pt")
                with torch.no_grad():
                    chunk_embedding = model(**chunk_tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
                chunk_embeddings.append(chunk_embedding)
            avg_embedding = np.mean(chunk_embeddings, axis=0)
            embeddings.append(avg_embedding)
        else:
            with torch.no_grad():
                embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)

    return np.mean(embeddings, axis=0)

def chunk_text_for_sentiment(text, max_length=510):
    """Helper function to split text into chunks for sentiment analysis."""
    tokens = tokenizer(text, truncation=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokenizer.decode(tokens[i:i + max_length])
        chunks.append(chunk)
    return chunks

def analyze_sentiment(text_data):
    """Analyze sentiment by chunking long texts."""
    sentiments = []
    for text in text_data:
        chunks = chunk_text_for_sentiment(text)
        for chunk in chunks:
            sentiment = sentiment_pipeline(chunk)
            sentiments.append(sentiment[0]["score"])
    return sentiments

def compute_weights(dates, engagement_scores, base_date=None):
    """Compute weights based on recency and engagement."""
    if not base_date:
        base_date = datetime.now()
    
    # Convert dates to days difference from base_date
    date_diffs = [(base_date - datetime.strptime(date, "%Y-%m-%d")).days for date in dates]
    
    # Normalize date differences and engagement scores
    date_diffs = MinMaxScaler().fit_transform(np.array(date_diffs).reshape(-1, 1)).flatten()
    engagement_scores = MinMaxScaler().fit_transform(np.array(engagement_scores).reshape(-1, 1)).flatten()
    
    # Higher weight for newer dates and higher engagement
    weights = (1 - date_diffs) + engagement_scores
    return weights / weights.sum()  # Normalize weights

def process_text_data_weighted(text_data, dates, engagement_scores, max_length=510):
    """Generate weighted embeddings using a pre-trained language model."""
    weights = compute_weights(dates, engagement_scores)
    embeddings = []
    
    for i, text in enumerate(text_data):
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding * weights[i])
    
    return np.sum(embeddings, axis=0)

def generate_contextual_embeddings(region_names, start_dates, end_dates):
    """Generate contextual embeddings for each region based on historical news, tweets, and Reddit posts."""
    contextual_embeddings = []
    region_subreddits = {
        "US": "CoronavirusUS",
        "IT": "italy",
        "ES": "spain",
        "EN": "CoronavirusUK",
        "FR": "CoronavirusFR"
    }

    for i, region in enumerate(region_names):
        # Initialize an empty list for text data from all sources
        texts = []
        dates = []
        engagement_scores = []
        avg_sentiments = []
        subreddit_name = region_subreddits.get(region, "Coronavirus")

        # Fetch news data
        news_data = fetch_contextual_data(
            GNEWS_API_KEY, "COVID-19", region, source='GNews',
            start_date=start_dates[i], end_date=end_dates[i]
        )
        if news_data:
            news_texts = [item['title'] for item in news_data.get('articles', [])]
            texts.extend(news_texts)
            dates.extend([start_dates[i]] * len(news_texts))
            engagement_scores.extend([1] * len(news_texts))  # Default engagement score for news
            avg_sentiments.extend(analyze_sentiment(news_texts))

        # Fetch Twitter data
        tweet_data = fetch_contextual_data(
            TWITTER_BEARER_TOKEN, "COVID-19", region, source='Twitter',
            start_date=start_dates[i], end_date=end_dates[i]
        )
        if tweet_data:
            tweet_texts = [tweet['text'] for tweet in tweet_data.get('data', [])]
            texts.extend(tweet_texts)
            dates.extend([start_dates[i]] * len(tweet_texts))
            engagement_scores.extend([1] * len(tweet_texts))  # Default engagement score for tweets
            avg_sentiments.extend(analyze_sentiment(tweet_texts))

        # Fetch Reddit data
        reddit_posts = fetch_reddit_data(subreddit_name, post_type="top", limit=30)
        if reddit_posts:
            reddit_texts = [post['title'] + ' ' + post['body'] for post in reddit_posts]
            texts.extend(reddit_texts)
            dates.extend([start_dates[i]] * len(reddit_texts))
            avg_sentiments.extend(analyze_sentiment(reddit_texts))

            # Gather engagement metrics
            avg_upvotes = np.mean([post.get('upvotes', 0) for post in reddit_posts])
            avg_comments = np.mean([post.get('num_comments', 0) for post in reddit_posts])
            engagement_scores.extend([avg_upvotes + avg_comments] * len(reddit_texts))

        # Generate weighted embeddings for the combined text data
        if texts:
            region_embedding = process_text_data_weighted(texts, dates, engagement_scores)
            avg_sentiment_score = np.mean(avg_sentiments) if avg_sentiments else 0

            # Append sentiment and engagement metrics
            region_embedding = np.append(region_embedding, [avg_sentiment_score, np.mean(engagement_scores), len(texts)])
        else:
            region_embedding = np.zeros(768 + 3)  # Assuming 768-dim for BERT embedding + 3 for metrics

        contextual_embeddings.append(region_embedding)

    # Scale embeddings and return
    scaler = MinMaxScaler()
    contextual_embeddings = scaler.fit_transform(contextual_embeddings)
    return contextual_embeddings
