"""
Copyright (c) 2020, Heung Kit Leslie Chung
All rights reserved.
Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""

from boilerpy3 import extractors
import spacy
import requests
from spacy.lang.en import English
from nltk.corpus import wordnet as wn
import pandas as pd
# pd.set_option("max_columns", None)
from textblob import TextBlob

# Use Spacy package model
nlp = spacy.load("en_core_web_md")
parser = English()

# Topic are the topics of interest - user should first decide which topics to be included for analysis
topics = ["automotive", "travel", "science", "technology",
          "nature", "sports", "business", "beauty", "fashion", "music", "art",
          "movie", "entertainment", "media", "children", "health", "gambling",
          "religion", "education", "politics", "history", "war", "news",
          "food", "literature", "crime", "analysis", "technical", "incident", "postmortem"]

# Function for removing irrelevant parts of the extracted text


def remove_trash(text, min_size=4):
    """

    Your input is a string. The string will be parsed by the character
    '\n' and any resulting component under 4 words long will be discarded.
    Returns user input text cleaned.

    Parameters
    ----------
    text : A string.

    Returns
    -------
    cleaned : A string.

    """
    cleaned = ''
    for i in text.split('\n'):
        if len(i.split()) <= min_size:
            continue
        else:
            cleaned += (i)
    return cleaned

# Function for tokenising and lammetization of the relevant text


def to_token(text):
    """

    Your input is a string. The string will be stripped off anything that looks
    like tagging, hashtags, web links, punctuations, numbers and stop words.
    Then the words will be lemmatised.

    Parameters
    ----------
    text : A string.

    Returns
    -------
    token_list : A list.

    """
    token_list = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:  # skip if looks like url
            continue
        elif token.orth_.startswith('@'):  # skip if looks like tag
            continue
        elif token.orth_.startswith('#'):  # skip if looks like tag
            continue
        elif token.is_punct:  # skip if looks like punctuation
            continue
        elif token.like_num:  # skip if looks like number
            continue
        elif token.is_stop:  # skip if looks like stop words
            continue
        else:
            token_list.append(token.lower_)

        token_list = [wn.morphy(x) if wn.morphy(
            x) != None else x for x in token_list]
    return token_list


def sentiment(text):
    """
    Your input is a string. The function will invoke the TextBlobk package's
    sentiment function and returns values of polarity and subjectivity to user.

    Parameters
    ----------
    text : A string.

    Returns
    -------
    A textblob.en.sentiments.Sentiment object.

    """
    return TextBlob(text).sentiment

# Function that takes url and topics (long string of single word topics separated by space) and return topic relevancy scores


def classify_web(url, topics=topics, analyse_sentiment=False):
    """
    This function enables users to take any url and return measures of topic
    similarity and sentiment analysis. The only required input is the url.
    Topics can be customised by the user and is taken by the function as a
    list of strings. If analyse_sentiment is set to true, the function will
    also return measures of polarity and subjectivity to the user.

    Parameters
    ----------
    url : A string.

    topics : A list, optional

    analyse_sentiment : Boolean, optional

    Returns
    -------
    A Pandas DataFrame containing topics similarity results and a
    textblob.en.sentiments.Sentiment object containing measures of polarity
    and subjectivity.

    """
    # Use boilerpy3 ArticleExtractor()
    extractor = extractors.ArticleExtractor()

    # Extract content from web url
    resp = requests.get(url)

    # content = extractor.get_content_from_url(url)
    content = extractor.get_content(resp.content.decode('utf-8'))

    # Join the tokens back up for Spacy nlp function
    token_join = ' '.join(to_token(remove_trash(content)))

    # doc is the content
    doc = nlp(token_join)

    # topics are the user defined topics
    topics = ' '.join(topics)
    topic = nlp(topics)

    # Calculate similarity between content and topics
    topic_similarity = []
    for token in topic:
        topic_similarity.append(nlp(token_join).similarity(token))

    # Calculate polarity (measure of positivity) and subjectivity
    senti_result = None
    if analyse_sentiment == True:
        senti_result = sentiment(content)

    # Table to capture results
    data = {'topics': topics.split(), 'similarity': topic_similarity}
    result = pd.DataFrame(data=data).sort_values(
        ['similarity'], ascending=[False]).reset_index(drop=True)

    return (result, senti_result)


if __name__ == "__main__":
    classify_web(
        "https://blog.cloudflare.com/1-1-1-1-lookup-failures-on-october-4th-2023/")
