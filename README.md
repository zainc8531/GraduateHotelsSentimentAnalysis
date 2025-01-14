# Graduate Hotels Sentiment Analysis
This project analyzes social media sentiment and customer feedback about Graduate Hotels by processing tweets, categorizing them into topics (like wifi, rooms, food, pricing), and visualizing the positive and negative sentiments to understand customer satisfaction and areas for improvement.


# Data Loading and Preprocessing:
The code loads tweet data from a JSON file called 'GraduateHotelCongregated.json'
It filters the data to keep only the id, content, and date columns
The code cleans the tweets by removing emojis, hashtags, URLs, mentions, and other Twitter-specific elements


# Topic Analysis:
The code creates several topic categories to analyze:


Wifi/Internet related tweets
Room/amenities related tweets
Food/restaurant related tweets
Price related tweets
Keywords about local culture/atmosphere
"Hallpass" related tweets (seems to be a specific feature/service)


# Sentiment Analysis:
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
Calculates compound scores (C_Score) for each tweet
Classifies tweets as Positive (>0.05), Neutral (-0.05 to 0.05), or Negative (<-0.05)
Creates visualizations (histograms and pie charts) showing sentiment distribution for each topic


# Word Cloud Visualization:
For each topic category, creates word clouds showing:
Most common words in positive tweets
Most common words in negative tweets


# Advanced Topic Modeling:
Uses BERTopic (a transformer-based topic modeling technique) to:
Automatically discover topics in the tweets
Visualize topic relationships
Create interactive visualizations of topic distributions
Save and load the topic model for future use
