

import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


api_key = 'a9cd5b8a108c43daae151e8ce8eec923' 
query = 'stock market'

url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={api_key}'
response = requests.get(url)
data = response.json()

analyzer = SentimentIntensityAnalyzer()
results = []

for article in data.get('articles', []):
    title = article.get('title', '')
    if title:
        sentiment = analyzer.polarity_scores(title)
        results.append({
            'title': title,
            'score': sentiment['compound']
        })

df = pd.DataFrame(results)
print(df)
