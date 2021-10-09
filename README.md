# A simple employee feedback Sentiment Analysis

This repository provides a simple sentiment prediction tool for employee fedback in a company. The current implementation was done in a simple machine without GPU, so beginners out there, you too can use this.

### How to Use
Command Line arguments -
```shell
~\sentiment.py -h
usage: sentiment.py [-h] [--i Input_path] [--co Output_1_path] [--so Output_2_path]

Analyze the sentiment of employee feedbacks

optional arguments:
	-h, --help show this help message and exit
	--i Input_path the path to input csv file
	--co Output_1_path the path to combined output csv file
	--so Output_2_path the path to sentence output csv file
	
Enjoy the program! :)
```
Example: 


    d:\> python main\sentiment.py --i .\train.csv --co .\c_out.csv --so .\s_out.csv

### Requirements
- tensorflow==2.6.0
- spacy==3.1.3
- spacy-legacy==3.0.8
- nltk==3.6.4
- numpy==1.19.5



### Synopsis
The challenge in natural language processing is to be able to extract meaning from sentences and paragraphs while focussing on a larger / macro view of the text otherwise the extracted meaning might turn out to be entirely different than the context. For the current execution, a simple custom model which is the basic step of ELMo algorithm due to computational constraints is used. Further a Bidirectional LSTM layer is used to extract information from the embeddings. Binary classification is considered where 0->1 translates to positive from negative.

#### Scoring
The repo extracts sentences and predicts the sentiment score for each of them, and a final result is taken by averaging the produced predictions to a range of 1.0 to 5.0



Dataset Used: [Hackerearth Challenge Dataset](https://www.hackerearth.com/challenges/hiring/ericsson-ml-challenge-2019/machine-learning/abc-test/ "Hackerearth Challenge Dataset")






#### Further Reading
- [ELMo](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/ "ELMo")
- [Bidirectional LSTM](https://analyticsindiamag.com/complete-guide-to-bidirectional-lstm-with-python-codes/ "Bidirectional LSTM")
- [Sentiment Analysis Concept](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17 "Sentiment Analysis Concept")

