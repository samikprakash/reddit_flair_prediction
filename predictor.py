import praw
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import string
import re
import random
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
# from func import Print
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
# from app.models import Link


def clean(text):
    stop_words = set(stopwords.words("english"))
    
    token = word_tokenize(text)
    token = [w.lower() for w in token]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in token]
    words = [word for word in stripped if word.isalpha()]
    impowrods = []
    for w in token:
        if w not in stop_words:
            impowrods.append(w)
    ps = PorterStemmer()
    wordnet = WordNetLemmatizer()
    stemmed = [wordnet.lemmatize(word) for word in impowrods]
    final = " ".join(stemmed)
    final = re.sub('W+','',final)
    return final


def predict(urls):

    infile = open("predictor.pickle","rb")
    nb = pickle.load(infile)
    infile.close()

    xtrain = open("xtrain.pickle","rb")
    X_train = pickle.load(xtrain)
    xtrain.close()

    ytrain = open("ytrain.pickle","rb")
    y_train = pickle.load(ytrain)
    ytrain.close()

    reddit = praw.Reddit(client_id ='6nyfvr2Ld0ZpqA',client_secret='m8JXMrtdZBpbptKU11uof9q4TPY',
    password='##########',username='samik2603',user_agent='mycoolapp v1 by /u/samik2603')

    subreddit = reddit.subreddit('india')
    # links = list(Link.objects.values_list())
    # link = links[-1][1]
    # link = link.replace(" ","")
    url = praw.models.Submission(reddit,url=urls)
    flair = (url.link_flair_text)
    X_test = []
    url.comments.replace_more(limit=0)
    comments = []
    for comment in url.comments.list():
        temp = list(comment.body.split())
        temp1 = " ".join(temp)
        comments.append(temp1)



    # print(comments)
    a = " ".join(comments)

    a = a+" " + url.title
    # print(a)
    X_test.append(a)
    X_test.append(flair)
    # print(X_test)



    X_test[0] = clean(X_test[0])
    wordnet = WordNetLemmatizer()
    # print(word_tokenize(wordnet.lemmatize(X_test[0])))
    # print(X_test)
    f = X_test[1]
    twod = []
    twod.append(X_test)
    dataset = pd.DataFrame(twod)

    dataset.columns = ["text","flair"]
    # print(dataset)

    # nb.fit(X_train,y_train)

    y_pred = nb.predict(dataset.iloc[:,0].values)
    Print(y_pred[0],f)

    return
