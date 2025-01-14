
#**0. Pre-step**
"""

# Commented out IPython magic to ensure Python compatibility.

from google.colab import drive
drive.mount('/content/drive')

# %cd /content/drive/MyDrive/488Project3

!ls

"""Loading in scrape data:"""

import pandas as pd

tweets = pd.read_json('GraduateHotelCongregated.json', lines=True)
pd.set_option('max_colwidth', 20)
tweets.tail()

# 1. Keep only certain columns
tweets = tweets.filter(['id','content','date'], axis=1)

# 2. We want to see as much of the columns as possible:
pd.set_option('max_colwidth', 400)

# 3. Let's take a look
display(tweets.head(10))
print()
display(tweets.info())

"""#**1. Filtering the data**

First we need to identify the emojis as they can affect the analysis. These two code blocks may need to be run inidivuidually as they can cause errors when running the whole notebook at once.
"""

# 1. First download and install the package
!pip install -U emoji

# 1. First download and install the package
!pip install emoji

"""#### **Filtering out emojis**"""

# 2. Import the packages we need
import emoji
import regex
import matplotlib.pyplot as plt

# 2. Define a function to get emojis
'Function that extracts the emojis from a text and returns them in a list'
def get_emojis(text):
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI["en"] for char in word):
            emoji_list.append(word)
    return emoji_list

# 3. Call function and pass it a Tweet
emoji_list = get_emojis(tweets.content[91])
print(' '.join(e for e in emoji_list))

# 4. Let's do that for all of our tweets and store the emojis in a new column called "emojis"
tweets['emojis'] = tweets['content'].apply(get_emojis)

# 6. Check if it worked
display(tweets) # logical indexing: from tweets get rows 340,352

"""Using Regex to remove hashtags, links, and mentions that may affect the analysis as they are twitter specific"""

# 1. Import regular expressions
import re

# 2. Set-up patterns to be removed fro the tweets
pat1 = r"http\S+"   # web links
pat2 = r"#"         # hashtags
pat3 = r"@"         # mentions
pat4 = r"FAV"       # twitter reserved abbreviation
pat5 = r"RE"        # twitter reserved abbreviation
pat6 = r"pic.\S+"   # twitter links to images
pat7 = r"\n"        # line breaks
pat8 = '\r\n'       # line breaks
pat9 = r'|'.join((r'&amp;',r'&copy;',r'&reg;',r'&quot;',r'&gt;',r'&lt;',r'&nbsp;',r'&apos;',r'&cent;',r'&euro;',r'&pound;'))  # HTML tags

# 3. Combine all patterns
combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8, pat9))

# 4. Replace the patterns with an empty string
tweets['stripped'] =  [re.sub(combined_pat, '', w) for w in tweets.content]

# 5. might have double spaces now (because of empty string replacements above) - remove double empty spaces
tweets['stripped'] = tweets.stripped.replace({' +':' '},regex=True)

# 6. Print some tweets to check if it worked
for i in range(0,10):
    print(tweets.stripped[i])
    print('\n')

"""[link text](https://)# Finding Frequency of Certain Topic Words

First, we need to search through the tweets for certain topic words that may be relevant for a sentiment analysis

#**2. Bag of words**

#### **Bag creations and counts**
"""

# 1. Import required modules (in case not already imported)
import numpy as np
import re

tweets['wifi'] = np.where(tweets.stripped.str.contains('(?:^|\W)(network|internet|wifi|web|cellular|connection)(?:$|\W)',
    flags = re.IGNORECASE), 1, 0)
tweets['room'] = np.where(tweets.stripped.str.contains('(?:^|\W)(room|bed|bathroom|TV|Television|amenities|temperature)(?:$|\W)',
    flags = re.IGNORECASE), 1, 0)
tweets['food'] = np.where(tweets.stripped.str.contains('(?:^|\W)(menu|taste|fresh|ingredients|restaurants|bars|food)(?:$|\W)',
    flags = re.IGNORECASE), 1, 0)
tweets['price'] = np.where(tweets.stripped.str.contains('(?:^|\W)(price|overpriced|expensive|cheap|affordable|rate)(?:$|\W)',
    flags = re.IGNORECASE), 1, 0)
tweets['keywords'] = np.where(tweets.stripped.str.contains('(?:^|\W)(local|collegiate culture|charm|nostalgic|friendly)(?:$|\W)',
    flags = re.IGNORECASE), 1, 0)

#when I observed wifi, I noticed a few about something called hall pass. So I did a separate search.
tweets['hallpass'] = np.where(tweets.stripped.str.contains('(?:^|\W)(hallpass|hall pass)(?:$|\W)',
    flags = re.IGNORECASE), 1, 0)


# 4. How many tweets of each topic?
print(f"Total {tweets['stripped'].count()}")
print(f"Wifi {tweets['wifi'].sum()}")
print(f"Room {tweets['room'].sum()}")
print(f"Food {tweets['food'].sum()}")
print(f"Price {tweets['price'].sum()}")
print(f"Keywords {tweets['keywords'].sum()}")
print(f"Hallpass {tweets['hallpass'].sum()}")
print()

"""#### **Examining Tweets**"""

def work(label, select_tweets):
  for w in select_tweets[0:10]:
    print(label + ':' + '\t' + w)

tweets[['stripped','wifi', 'room', 'food', 'price', 'keywords', 'hallpass']].head(10)
select_tweets = tweets.loc[tweets['wifi'] == 1, 'stripped'].values[:]
work("Wifi", select_tweets)
print()
select_tweets = tweets.loc[tweets['room'] == 1, 'stripped'].values[:]
work("Room", select_tweets)
print()

select_tweets = tweets.loc[tweets['food'] == 1, 'stripped'].values[:]
work("Food", select_tweets)
print()

select_tweets = tweets.loc[tweets['price'] == 1, 'stripped'].values[:]
work("Price", select_tweets)
print()

select_tweets = tweets.loc[tweets['keywords'] == 1, 'stripped'].values[:]
work("Keywords", select_tweets)
print()

select_tweets = tweets.loc[tweets['hallpass'] == 1, 'stripped'].values[:]
work("Hallpass", select_tweets)

"""#**3. Word Clouds**

#### **Wifi**
"""

from wordcloud import WordCloud

all_words = ' '.join([text for text in tweets[(tweets['wifi'] == 1)]['stripped']])
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#### **Room**"""

all_words = ' '.join([text for text in tweets[(tweets['room'] == 1)]['stripped']])
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)


# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#### **Food**"""

all_words = ' '.join([text for text in tweets[(tweets['food'] == 1)]['stripped']])
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#### **Price**"""

all_words = ' '.join([text for text in tweets[(tweets['price'] == 1)]['stripped']])
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#### **Keywords**"""

all_words = ' '.join([text for text in tweets[(tweets['keywords'] == 1)]['stripped']])
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#### **Hallpass**"""

all_words = ' '.join([text for text in tweets[(tweets['hallpass'] == 1)]['stripped']])
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**4. Sentiment Analysis Pre-Step**"""

# 0. Run once to install the Vader Sentiment Classification Package
!pip install vaderSentiment

"""####**C_Score Calculation**"""

# 1. Import the sentiment module (in case you haven't already done so)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 2. Import numpy (in case you have not already done so)
import numpy as np

# 3. Instantiate the sentiment analyzer (in case you haven't already done so)
analyser = SentimentIntensityAnalyzer()

# 4. Now get the compound sentiment score for each tweet
tweets['C_Score'] = np.nan # initialize empty comlumn in our tweets dataframe (empty = missing values)
for index, row in tweets.iterrows():  # loop through all tweets (i.e., rows)
    tweets.loc[index, 'C_Score'] = analyser.polarity_scores(row['stripped'])['compound']

# 5. Let's take a look!
pd.set_option('display.max_colwidth', None)
tweets[['stripped','C_Score']][0:1000]

"""#### **Tweet C_Score**"""

# 1. import necessary modules (in case not already imported)
import pandas as pd
import numpy as np

print(f"Count positive tweets: {sum(tweets['C_Score'] > 0.05)}")
print(f"Count netural tweets: {tweets['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets['C_Score'].count()}")
print()
display(tweets.C_Score.describe())

"""#**5. Wifi Sentiment Analysis**"""

print(f"Count positive tweets: {sum(tweets[(tweets['wifi'] == 1)]['C_Score']> 0.05)}")
print(f"Count netural tweets: {tweets[(tweets['wifi'] == 1)]['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets[(tweets['wifi'] == 1)]['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets[(tweets['wifi'] == 1)]['C_Score'].count()}")

# 1. import necessary modules (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Settings for seaborn plotting style
sns.set(color_codes=True)

# 3. Settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

# 4. Create Histogram
ax = sns.histplot(tweets[(tweets['wifi'] == 1)]['C_Score'],
                  bins=10,
                  kde=False,
                  color='skyblue')
ax.set(xlabel='Sentiment Distribution for Wifi', ylabel='Frequency')

# 1. Create new column with missing values
tweets['Sentiment'] = np.nan

# 2. Loop through rows of dataframe and determine strings for new column "Sentiment"
for index, row in tweets.iterrows():
    if tweets.loc[index, 'C_Score'] > 0.05 :
            tweets.loc[index, 'Sentiment'] = "Positive"
    elif tweets.loc[index, 'C_Score'] < -0.05 :
            tweets.loc[index, 'Sentiment'] = "Negative"
    else :
        tweets.loc[index, 'Sentiment'] = "Neutral"

# 3. Typecast as categorical variable (computationally more efficient)
tweets['Sentiment'] = tweets['Sentiment'].astype("category")

"""####**Sentiment Distribution**"""

# 1. Import necessary modules (in case not already imported)
import matplotlib.pyplot as plt

# 2. Set font size
plt.rcParams['font.size']=24

# 3. Define figure
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))

# 4. Get count by sentiment category from tweets_df

sentiment_counts = tweets[(tweets['wifi'] == 1)].Sentiment.value_counts()
labels = sentiment_counts.index

# 5. Define colors
color_palette_list = ['lightgreen', 'red', 'lightblue','orange']

# 6. Generate graph components
wedges, texts, autotexts = ax.pie(sentiment_counts, wedgeprops=dict(width=0.5), startangle=-40,
       colors=color_palette_list[0:3], autopct='%1.0f%%', pctdistance=.75, textprops={'color':"w", 'weight':'bold'})

# 7. Plot wedges
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(labels[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment)
# 8. Set title
ax.set_title("Sentiment Distribution", y=.95, fontsize = 24)

# 9. Show Doughnut Chart
plt.show()

"""####**Common Words in Positive Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Positive"  ) & (tweets['wifi'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""####**Common Words in Negative Tweets**

"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Negative"  ) & (tweets['wifi'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**6. Room Sentiment Analysis**"""

print(f"Count positive tweets: {sum(tweets[(tweets['room'] == 1)]['C_Score']> 0.05)}")
print(f"Count netural tweets: {tweets[(tweets['room'] == 1)]['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets[(tweets['room'] == 1)]['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets[(tweets['room'] == 1)]['C_Score'].count()}")

# 1. import necessary modules (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Settings for seaborn plotting style
sns.set(color_codes=True)

# 3. Settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

# 4. Create Histogram
ax = sns.histplot(tweets[(tweets['room'] == 1)]['C_Score'],
                  bins=10,
                  kde=False,
                  color='skyblue')
ax.set(xlabel='Sentiment Distribution for Rooms', ylabel='Frequency')

"""####**Sentiment Distribution**"""

# 1. Import necessary modules (in case not already imported)
import matplotlib.pyplot as plt

# 2. Set font size
plt.rcParams['font.size']=24

# 3. Define figure
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))

# 4. Get count by sentiment category from tweets_df

sentiment_counts = tweets[(tweets['room'] == 1)].Sentiment.value_counts()
labels = sentiment_counts.index

# 5. Define colors
color_palette_list = ['lightgreen', 'red', 'lightblue','orange']

# 6. Generate graph components
wedges, texts, autotexts = ax.pie(sentiment_counts, wedgeprops=dict(width=0.5), startangle=-40,
       colors=color_palette_list[0:3], autopct='%1.0f%%', pctdistance=.75, textprops={'color':"w", 'weight':'bold'})

# 7. Plot wedges
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(labels[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment)
# 8. Set title
ax.set_title("Sentiment Distribution", y=.95, fontsize = 24)

# 9. Show Doughnut Chart
plt.show()

"""####**Common Words in Positive Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Positive"  ) & (tweets['room'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""####**Common Words in Negative Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Negative"  ) & (tweets['room'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**7. Food Sentiment Analysis**"""

print(f"Count positive tweets: {sum(tweets[(tweets['food'] == 1)]['C_Score']> 0.05)}")
print(f"Count netural tweets: {tweets[(tweets['food'] == 1)]['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets[(tweets['food'] == 1)]['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets[(tweets['food'] == 1)]['C_Score'].count()}")

# 1. import necessary modules (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Settings for seaborn plotting style
sns.set(color_codes=True)

# 3. Settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

# 4. Create Histogram
ax = sns.histplot(tweets[(tweets['food'] == 1)]['C_Score'],
                  bins=10,
                  kde=False,
                  color='skyblue')
ax.set(xlabel='Sentiment Distribution for Food', ylabel='Frequency')

"""####**Sentiment Distribution**"""

# 1. Import necessary modules (in case not already imported)
import matplotlib.pyplot as plt

# 2. Set font size
plt.rcParams['font.size']=24

# 3. Define figure
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))

# 4. Get count by sentiment category from tweets_df

sentiment_counts = tweets[(tweets['food'] == 1)].Sentiment.value_counts()
labels = sentiment_counts.index

# 5. Define colors
color_palette_list = ['lightgreen', 'red', 'lightblue','orange']

# 6. Generate graph components
wedges, texts, autotexts = ax.pie(sentiment_counts, wedgeprops=dict(width=0.5), startangle=-40,
       colors=color_palette_list[0:3], autopct='%1.0f%%', pctdistance=.75, textprops={'color':"w", 'weight':'bold'})

# 7. Plot wedges
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(labels[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment)
# 8. Set title
ax.set_title("Sentiment Distribution", y=.95, fontsize = 24)

# 9. Show Doughnut Chart
plt.show()

"""####**Common Words in Positive Tweets**"""

# 1. Import module
from wordcloud import WordCloud


# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Positive"  ) & (tweets['food'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""####**Common Words in Negative Tweets**"""

# 1. Import module
from wordcloud import WordCloud


# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Negative"  ) & (tweets['food'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**8. Price Sentiment Analysis**"""

print(f"Count positive tweets: {sum(tweets[(tweets['price'] == 1)]['C_Score']> 0.05)}")
print(f"Count netural tweets: {tweets[(tweets['price'] == 1)]['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets[(tweets['price'] == 1)]['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets[(tweets['price'] == 1)]['C_Score'].count()}")

# 1. import necessary modules (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Settings for seaborn plotting style
sns.set(color_codes=True)

# 3. Settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

# 4. Create Histogram
ax = sns.histplot(tweets[(tweets['price'] == 1)]['C_Score'],
                  bins=10,
                  kde=False,
                  color='skyblue')
ax.set(xlabel='Sentiment Distribution for Price', ylabel='Frequency')

"""####**Sentiment Distribution**"""

# 1. Import necessary modules (in case not already imported)
import matplotlib.pyplot as plt

# 2. Set font size
plt.rcParams['font.size']=24

# 3. Define figure
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))

# 4. Get count by sentiment category from tweets_df

sentiment_counts = tweets[(tweets['price'] == 1)].Sentiment.value_counts()
labels = sentiment_counts.index

# 5. Define colors
color_palette_list = ['lightgreen', 'red', 'lightblue','orange']

# 6. Generate graph components
wedges, texts, autotexts = ax.pie(sentiment_counts, wedgeprops=dict(width=0.5), startangle=-40,
       colors=color_palette_list[0:3], autopct='%1.0f%%', pctdistance=.75, textprops={'color':"w", 'weight':'bold'})

# 7. Plot wedges
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(labels[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment)
# 8. Set title
ax.set_title("Sentiment Distribution for Price", y=.95, fontsize = 24)

# 9. Show Doughnut Chart
plt.show()

"""####**Common Words in Positive Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Positive"  ) & (tweets['price'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""####**Common Words in Negative Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Negative"  ) & (tweets['price'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**9. Keywords Sentiment Analysis**"""

print(f"Count positive tweets: {sum(tweets[(tweets['keywords'] == 1)]['C_Score']> 0.05)}")
print(f"Count netural tweets: {tweets[(tweets['keywords'] == 1)]['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets[(tweets['keywords'] == 1)]['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets[(tweets['keywords'] == 1)]['C_Score'].count()}")

# 1. import necessary modules (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Settings for seaborn plotting style
sns.set(color_codes=True)

# 3. Settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

# 4. Create Histogram
ax = sns.histplot(tweets[(tweets['keywords'] == 1)]['C_Score'],
                  bins=10,
                  kde=False,
                  color='skyblue')
ax.set(xlabel='Sentiment Distribution for keywords', ylabel='Frequency')

"""####**Sentiment Distribution**"""

# 1. Import necessary modules (in case not already imported)
import matplotlib.pyplot as plt

# 2. Set font size
plt.rcParams['font.size']=24

# 3. Define figure
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))

# 4. Get count by sentiment category from tweets_df

sentiment_counts = tweets[(tweets['keywords'] == 1)].Sentiment.value_counts()
labels = sentiment_counts.index

# 5. Define colors
color_palette_list = ['lightgreen', 'red', 'lightblue','orange']

# 6. Generate graph components
wedges, texts, autotexts = ax.pie(sentiment_counts, wedgeprops=dict(width=0.5), startangle=-40,
       colors=color_palette_list[0:3], autopct='%1.0f%%', pctdistance=.75, textprops={'color':"w", 'weight':'bold'})

# 7. Plot wedges
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(labels[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment)
# 8. Set title
#ax.set_title("Sentiment Distribution", y=.95, fontsize = 20)

# 9. Show Doughnut Chart
plt.show()

"""####**Common Words in Positive Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Positive"  ) & (tweets['keywords'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""####**Common Words in Negative Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Negative"  ) & (tweets['keywords'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**10. Hallpass Sentiment Analysis**"""

print(f"Count positive tweets: {sum(tweets[(tweets['hallpass'] == 1)]['C_Score']> 0.05)}")
print(f"Count netural tweets: {tweets[(tweets['hallpass'] == 1)]['C_Score'].between(-0.05, 0.05).sum()}")
print(f"Count negative tweets: {sum(tweets[(tweets['hallpass'] == 1)]['C_Score'] < -0.05)}")
print(f"Total number of tweets: {tweets[(tweets['hallpass'] == 1)]['C_Score'].count()}")

# 1. import necessary modules (in case not already imported)
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Settings for seaborn plotting style
sns.set(color_codes=True)

# 3. Settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})

# 4. Create Histogram
ax = sns.histplot(tweets[(tweets['hallpass'] == 1)]['C_Score'],
                  bins=10,
                  kde=False,
                  color='skyblue')
ax.set(xlabel='Sentiment Distribution for HallPass', ylabel='Frequency')

"""####**Sentiment Distribution**"""

# 1. Import necessary modules (in case not already imported)
import matplotlib.pyplot as plt

# 2. Set font size
plt.rcParams['font.size']=24

# 3. Define figure
fig, ax = plt.subplots(figsize=(9, 6), subplot_kw=dict(aspect="equal"))

# 4. Get count by sentiment category from tweets_df

sentiment_counts = tweets[(tweets['hallpass'] == 1)].Sentiment.value_counts()
labels = sentiment_counts.index

# 5. Define colors
color_palette_list = ['lightgreen', 'red', 'lightblue','orange']

# 6. Generate graph components
wedges, texts, autotexts = ax.pie(sentiment_counts, wedgeprops=dict(width=0.5), startangle=-40,
       colors=color_palette_list[0:3], autopct='%1.0f%%', pctdistance=.75, textprops={'color':"w", 'weight':'bold'})

# 7. Plot wedges
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    ax.annotate(labels[i], xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment)
# 8. Set title
ax.set_title("Sentiment Distribution", y=.95, fontsize = 24)

# 9. Show Doughnut Chart
plt.show()

"""####**Common Words in Positive Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Positive"  ) & (tweets['hallpass'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""####**Common Words in Negative Tweets**"""

# 1. Import module
from wordcloud import WordCloud

# 4. Create bag of words for tweets of certain sentiment
all_words = ' '.join([text for text in tweets[(tweets['Sentiment'] =="Negative"  ) & (tweets['hallpass'] == 1)]['stripped']])

# 5. Generate Word Cloud
wordcloud = WordCloud(collocations=True, width=800, height=500, random_state=5, max_font_size=110).generate(all_words)

# 6. Visulaize Cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

"""#**11. Bert**"""

!pip install bertopic

!pip install bertopic[visualization]

# Import libraries
import numpy as np
from bertopic import BERTopic

# 1. Instantiate model
model = BERTopic(verbose=True, nr_topics="auto")

# 2. Convert tweets to list as input to model
docs = tweets["stripped"].to_list()

 # 3. Fit model to data to predict topics
topics, probabilities = model.fit_transform(docs)

# Let's see how many tweets are in each discovered topic!
model.get_topic_freq().head(20)

# Get more details: The model even tries to give names to topics... well... sort of works... sort of...
model.get_topic_info()

# Find the words of a specific topic (and their probabilities of belonging to that topic)
model.get_topic(25)

# You can even interactively explore topics
model.visualize_topics()

# A nice feature of BERTopic is that you can easily generate Barcharts of the most relevant words per topic
model.visualize_barchart()

# You can create an interactive heatmap
model.visualize_heatmap()

# Define sentence
new_doc = "Had great food at the graduate hotel"
# Feed it into the model
model.transform([new_doc])
# should find that belongs to topic 1 ([1], None)

pd.DataFrame(model.find_topics("wifi"))
# most relevant is with highest score (row 1, and then column, which corresponds to topic number).

pd.DataFrame(model.find_topics("room"))
# most relevant is with highest score (row 1, and then column, which corresponds to topic number).

pd.DataFrame(model.find_topics("food"))
# most relevant is with highest score (row 1, and then column, which corresponds to topic number).

pd.DataFrame(model.find_topics("price"))
# most relevant is with highest score (row 1, and then column, which corresponds to topic number).

pd.DataFrame(model.find_topics("hallpass"))
# most relevant is with highest score (row 1, and then column, which corresponds to topic number).

# save model (we name the file "graduatetwees")
model.save("graduatetweets")

# load model (now we name the model "graduate_model")
graduate_model = BERTopic.load("graduatetweets")

# check loaded model
graduate_model.visualize_barchart()
