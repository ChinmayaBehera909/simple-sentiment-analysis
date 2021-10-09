def text_preprocessing(str_input): 
    
    import spacy
    import re

    #stop words list
    stopwords = ['I','a','about','an','are','as','at','be','by','com','for','from','how','in','is','it','of','on','or',
                 'that','the','this','to','was','what','when','where','who','will','with','the']

    nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])

    #tokenization, remove punctuation, lemmatization
    words=[token.lemma_ for token in nlp(str_input) if not token.is_punct]
 
    # remove symbols, websites, email addresses 
    words = [re.sub(r"[^A-Za-z@]", "", word) for word in words] 
    words = [re.sub(r"\S+com", "", word) for word in words]
    words = [re.sub(r"\S+@\S+", "", word) for word in words] 
    words = [word for word in words if word!=' ']
    words = [word for word in words if len(word)!=0] 
 
    #remove stopwords     
    words=[word.lower() for word in words if word.lower() not in stopwords]

    #combine a list into one string   
    string = " ".join(words)

    return string

def clean_and_extract(df, preprocess=False):

    from nltk import sent_tokenize
    # extract sentences from paragraph
    sentences=[]
    id=[]
    for k in range(len(df)):
        for s in sent_tokenize(df['Feedbacks'][k]):
            sentences.append(s)
            id.append(df['ID'][k])
    n_s = sentences.copy()
    ret_dict ={
        'ID': id,
        'Sentences:':n_s
    }

    # reduction process
    if preprocess:
        for s in range(len(sentences)):
            sentences[s] = text_preprocessing(sentences[s])
    
    return sentences, ret_dict

def predict_nlp(text, df):
    from tensorflow.keras.preprocessing.text import Tokenizer,tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    import os

    import json


    # load the bag or words
    with open(os.path.join(BASE_DIR,'tokenizer.json')) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    # model configuration constraints
    max_length = 200
    trunc_type='post'
    padding_type='post'

    # convert sentences to token sequences
    sequences = tokenizer.texts_to_sequences(text)
    # padding the sequences to equal length
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    # load the model
    model = load_model(os.path.join(BASE_DIR,'sentiment_sentence.h5'))
    # sentiment prediction
    pred_sentences = model.predict(padded)
    
    df['Sentiment_values'] = pred_sentences.reshape(pred_sentences.shape[0])
    
    return df

def predict_sentiment(dataframe):
    
    import numpy as np
    
    # clean up the text and extract sentences
    print('Started cleaning...',end='')
    text_sentences, dict_store = clean_and_extract(dataframe,True)
    print('Done')
    print('Started Prediction...',end='')
    dict_store = predict_nlp(text_sentences, dict_store)
    print('Done')
    dict_store['Sentiment_Scores'] = np.around(4*dict_store['Sentiment_values'])+1
    df = pd.DataFrame(dict_store)

    # model scoring
    predictions = []
    unique_id = np.unique(df['ID'])
    for id in unique_id:
        x = df.loc[df['ID']==id]
        predictions.append(np.mean(x['Sentiment_Scores']))
    
    dataframe['Sentiment_Score'] = predictions
    dataframe_sentence = pd.DataFrame(dict_store)
    return dataframe, dataframe_sentence


if __name__ == '__main__':

    import argparse
    import pandas as pd
    import os
    import sys

    global BASE_DIR
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Analyze the sentiment of employee feedbacks',
                                    epilog='Enjoy the program! :)', allow_abbrev=False)

    # Add the arguments
    my_parser.add_argument('--i',metavar='Input_path',
                        type=str,
                        help='the path to input csv file',)
    my_parser.add_argument('--co',metavar='Output_1_path',
                        type=str,
                        help='the path to combined output csv file',)
    my_parser.add_argument('--so',metavar='Output_2_path',
                        type=str,
                        help='the path to sentence output csv file')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.i
    out_combined = args.co
    out_sentence = args.so


    try:
        print('Reading input file...',end='')
        df = pd.read_csv(input_path)
        print('Done')
        try:
            df1, df2 = predict_sentiment(df)
        except Exception as ie:
            print('Internal error while predicting!', ie)
        
        try:
            df1.to_csv(out_combined)
            df2.to_csv(out_sentence)
        except Exception as ex:
            print(ex)
            sys.exit()

    except Exception as e:
        print(e)
        sys.exit()