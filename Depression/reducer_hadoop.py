import csv, sys
from transformers import pipeline
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Fine-tune the model on a labeled dataset
# (not shown here)

# Define a function to classify a given sentence
def classify_sentiment(sentence):
    # Tokenize the input sentence
    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Run the input through the fine-tuned model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    scores = torch.softmax(logits, dim=1).detach().numpy()[0]

    # Map the scores to positive or negative sentiment based on a threshold
    return scores[1]


# Open CSV file for writing results
df = pd.DataFrame(columns = ['Date', 'sentecne','sentiment_score' ])

# Loop through input from mapper
for line in tqdm(sys.stdin):
    # Parse input line
    line = line.strip()
    ln = line.split('\t')
    text = " ".join(ln[:-1])
    date = ln[-1]
    # Calculate sentiment and confidence score with BERT
    sentiment = classify_sentiment(text)
    df = df.append({
        'Date':date,'sentence':text,'sentiment_score':sentiment
    },ignore_index=True)
# calculate medTrue
median = df['sentiment_score'].median()

# replace values below median with negative and above median with positive
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Negative' if x < median else 'Positive')
df.to_csv('Sentiment_results.csv',index = True)
