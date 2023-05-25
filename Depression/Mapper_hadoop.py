from pymongo import MongoClient
import string
import re, sys
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import pipeline

# Define stop words and stemmer
#stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Connect to MongoDB and retrieve text field
client = MongoClient()
db = client['Twitter']
collection = db['RAW_DATA']
docs = collection.find()

for doc in docs:
    text = doc['Text']
    date = doc['Date']
    # Preprocess text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.findall(r'\b\w+\b', text)
    #words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(ps.stem(w)) for w in words]
    word = " ".join(words)
    sys.stdout.reconfigure(encoding='utf-8')
    print(word + "\t" + str(date) )
