from keras.models import model_from_json
import pickle
import numpy as np
import keras
import tensorflow as tf
import pickle
from textblob import TextBlob
from copy import deepcopy
from keras.preprocessing.text import Tokenizer
from sklearn import metrics
import back_end
from copy import deepcopy
import random
import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json,custom_objects={"keras":keras})
loaded_model.load_weights("weights.best.hdf5")
print("Loaded Model from file")
val_splt=0.1
#--------------------------------------------- Copy Respective Front_end
with open('risk_mapping.pkl', 'rb') as f:
    risk_data = pickle.load(f)
def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)

def unison_shuffled_copies(a, b):
    #assert len(a) == len(b)
    a=np.array(a)
    p = np.random.permutation(a.shape[1])
    for i in range(a.shape[0]):
        a[i]=deepcopy(a[i][p]);
    b=deepcopy(b[p])
    return a.tolist(), b

with open('input_data_test', 'rb') as f:
    input_data = pickle.load(f)
    
with open('narrative_data_test', 'rb') as f:
    narrative_data_main = pickle.load(f)    

with open('risk_encoding', 'rb') as f:
    encoded_risk = pickle.load(f)
    
#Convert Pd to list

#input_data=input_data[:500,:]

#input_data=input_data.values.tolist()

# Treat Narrative Data like any other input
#input_data.append(narrative_data_main)

#input_data,encoded_risk=unison_shuffled_copies(input_data,encoded_risk)

slice_v=len(input_data[0]) # To use full dataset, replace 5000 with len(input_data[0])

for i in range(len(input_data)):
    input_data[i]=input_data[i][:slice_v]
#narrative_data_main=narrative_data_main[:slice_v]
encoded_risk=encoded_risk[:slice_v,:]

num_input_columns=len(input_data);

col=[];

for i_inp in range(num_input_columns):
    if i_inp==2:  # Hard Coded value: colum corresponding to crew size
        continue;
    col.append(input_data[i_inp])




integer_inps=input_data[2];

#print(len(input_data[0]))
#print(len(input_data[1]))
#print(len(input_data[2]))
#print(len(input_data[3]))
#print(len(input_data[4]))
#print(len(narrative_scores))
#print(encoded_risk.shape)




# Initialize tokenizers
#-------------------

num_string_inps=len(col);

tokenizers=[]
for i_token in range(num_string_inps):
    tokenizers.append(Tokenizer())

#------------------

# Fit tokenizers, to categorical data

for i_token in range(num_string_inps):

    tokenizers[i_token].fit_on_texts(col[i_token])
# Get the vocabulary size learned for each input channel
vocab_sizes=[]

for i_token in range(num_string_inps):
    vocab_sizes.append( len(tokenizers[i_token].word_index) + 1)

print(vocab_sizes)
#------------------
# Pre-trained embedding for narrative

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_sizes[-1], 100))
for word, i in tokenizers[-1].word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


#-------------------------------
# Create sequence for each input, based on fits, and pad with zeros for input to LSTM
encoded_col=[];
for i_token in range(num_string_inps):
    maxlen = find_max_list(list(map(str.split,col[i_token])))
    
    temp_list=tokenizers[i_token].texts_to_sequences(col[i_token])

    for i in range(len(temp_list)):
        temp_list[i]=temp_list[i]+[0] * (maxlen - len(temp_list[i])) # Pad with 0s to get consistent input size
    encoded_col.append(np.expand_dims(np.array(temp_list),axis=2))
#    encoded_col.append(np.array(temp_list))
    print(encoded_col[i_token].shape)
#--------------------------------:

#print(encoded_col[0][1000,:])
integer_inps=np.array(integer_inps)
#narrative_scores=np.array(narrative_scores)
integer_inps=np.expand_dims(integer_inps,axis=1)
#narrative_scores=np.expand_dims(narrative_scores,axis=1)
#numerical_inputs=np.concatenate((integer_inps,narrative_scores),axis=1)


encoded_col.append(np.expand_dims(integer_inps,axis=2))
#print(encoded_col[4].shape)

input_list=deepcopy(encoded_col)
output_list=deepcopy([encoded_risk])
#----------------------------------------------------------------

ans=loaded_model.predict(input_list)


val_slice=int(val_splt*len(input_data[0]))
val_out=ans[-val_slice:,:]
risk_data=encoded_risk[-val_slice:,:]

int_vals_out=[]
int_vals_true=[]
for i in range(val_out.shape[0]):

    int_vals_out.append(np.argmax(val_out[i,:])+1)
    int_vals_true.append(np.argmax(risk_data[i,:])+1)

int_vals_out=np.array(int_vals_out)

matrix=metrics.confusion_matrix(y_true=int_vals_true,y_pred=int_vals_out)

recall_scores=[]

for i in range(matrix.shape[0]):

    recall_scores.append(float(matrix[i,i])/sum(matrix[i,:]))
print(recall_scores)


df_cm = pd.DataFrame(matrix)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("GLoVE Prediction")
plt.show()
