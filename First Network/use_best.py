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
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("weights.best.hdf5")
print("Loaded Model from file")

#--------------------------------------------- Copy Respective Front_end

val_splt=0.10

def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)

with open('risk_mapping.pkl', 'rb') as f:
    risk_data = pickle.load(f)

with open('input_data_test', 'rb') as f:
    input_data = pickle.load(f)

with open('narrative_data_test', 'rb') as f:
    narrative_scores = pickle.load(f)

with open('risk_encoding', 'rb') as f:
    encoded_risk = pickle.load(f)

slice_v=int(len(input_data[0]))

for i in range(len(input_data)):
    input_data[i]=input_data[i][:slice_v]
narrative_scores=narrative_scores[:slice_v]
encoded_risk=encoded_risk[:slice_v,:]
num_input_columns=len(input_data)

col=[];

for i_inp in range(num_input_columns):
    if i_inp==2:  # Hard Coded value: colum corresponding to crew size
        continue;
    col.append(input_data[i_inp])

integer_inps=input_data[2];

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


#-------------------------------
# Create sequence for each input, based on fits, and pad with zeros for input to LSTM
encoded_col=[];
for i_token in range(num_string_inps):
    maxlen = find_max_list(list(map(str.split,col[i_token])))
    
    temp_list=tokenizers[i_token].texts_to_sequences(col[i_token])

    for i in range(len(temp_list)):
        temp_list[i]=temp_list[i]+[0] * (maxlen - len(temp_list[i])) # Pad with 0s to get consistent input size
    encoded_col.append(np.expand_dims(np.array(temp_list),axis=2))
    print(encoded_col[i_token].shape)
#--------------------------------:

#print(encoded_col[0][1000,:])
integer_inps=np.array(integer_inps)
narrative_scores=np.array(narrative_scores)
integer_inps=np.expand_dims(integer_inps,axis=1)
narrative_scores=np.expand_dims(narrative_scores,axis=1)
numerical_inputs=np.concatenate((integer_inps,narrative_scores),axis=1)


encoded_col.append(np.expand_dims(numerical_inputs,axis=2))
#print(encoded_col[4].shape)

input_list=deepcopy(encoded_col)
output_list=deepcopy([encoded_risk])

#----------------------------------------------------------------

ans=loaded_model.predict(input_list)

val_slice=int(val_splt*len(input_data[0]))
val_out=ans[-val_slice:,:]
val_true=encoded_risk[-val_slice:,:]

class_size=np.zeros([5])



int_vals_pred=[]
int_vals_true=[]
for i in range(val_out.shape[0]):

    int_vals_pred.append(np.argmax(val_out[i,:])+1)
    int_vals_true.append(np.argmax(val_true[i,:])+1)

int_vals_true=np.array(int_vals_true)
int_vals_pred=np.array(int_vals_pred)


matrix=metrics.confusion_matrix(y_true=int_vals_true,y_pred=int_vals_pred)

recall_scores=[]

for i in range(matrix.shape[0]):

    recall_scores.append(float(matrix[i,i])/sum(matrix[i,:]))
print(recall_scores)


#print(matrix)
df_cm = pd.DataFrame(matrix)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}) # font size
plt.xlabel("Predicted Labels")
plt.ylabel("true Labels")
plt.title("Sentiment Analysis")
plt.show()
