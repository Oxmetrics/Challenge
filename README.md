# Challenge Solution

This solution was influenced by https://github.com/linanqiu/word2vec-sentiments and results in a code that combines the doc2vec
model with a sentiment classification algorithm in Python class SentimentModel.

## Assumptions and Data
It is assumed that all three documents (positive, negative and unsupervised reviews) consist of sentences separated by new lines.
In our example, negative_reviews.txt, positive_reviews.txt and unsupervised_reviews.txt have 25000, 25000 and 50000 thus identified sentences.

## Code
There are three classes in challenge.py. DocDescription is simply an auxiliary class that provides a description of the document 
with labelled sentences. LabeledDoc reads the data and splits it into labelled sentences. It also counts the overall number of sentences
and separates them into training and testing sets using labels. SentimentModel is the main class that combines the doc2vec model with
sentiment classification and analysis. It takes in three labelled documents that represent negative, positive and unsupervised reviews. After some standard manipulation the classification accuracy level can be outputted to the console.


## Running the Code
Simply run challenge.py. Feel free to fiddle with parameters and other inputs.
