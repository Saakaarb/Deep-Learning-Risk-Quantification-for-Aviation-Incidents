#!/usr/bin/env python
# coding: utf-8

# In[1]:


# test network (5 inputs)

#tokenize
#embedding matrix-one hot encoding
#pass to an RNN

import numpy as np
import keras
import tensorflow as tf
import pickle
from textblob import TextBlob
from copy import deepcopy
from keras.preprocessing.text import Tokenizer
import back_end
from copy import deepcopy
import random

def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)
'''
def unison_shuffled_copies(a,b,c):
    #assert len(a) == len(b)
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    
    p = np.random.permutation(a.shape[1])
    for i in range(a.shape[0]):
        a[i]=deepcopy(a[i][p]);
    b=deepcopy(b[p])
    c=deepcopy(c[p])
    return a.tolist(), b.tolist(),c

'''
with open('input_data_test', 'rb') as f:
    input_data = pickle.load(f)
    
with open('narrative_data_test', 'rb') as f:
    narrative_scores = pickle.load(f)    

with open('risk_encoding', 'rb') as f:
    encoded_risk = pickle.load(f)
    
#Convert Pd to list

#input_data=input_data[:500,:]


#input_data=input_data.values.tolist()

#input_data,narrative_scores,encoded_risk=unison_shuffled_copies(input_data,narrative_scores,encoded_risk)

slice_v=len(input_data[0])

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
#-----------------------------

# Create the model

exp_no=1;


num_input_channels=num_string_inps
num_LSTM_layers=[1,1,1,1];
num_LSTM_units=[[64],[64],[64],[64]];
num_dense_layers=3;
num_dense_units=[256,128,5]; # Last dense units always has to be number of risk categories (5)
learning_rate=10**(-3);
inp_optim="Adam";
reg_param=10**(-4);
batch_sz=2048;
eps=500;
val_splt=0.15;



input_shapes=[]
for i_inp in range(num_input_channels):

    input_shapes.append(encoded_col[i_inp].shape)
#print(input_shapes[1][1])

model=back_end.create_model(num_input_channels,input_shapes,num_LSTM_layers,num_LSTM_units,num_dense_layers,num_dense_units,learning_rate,reg_param,inp_optim)

back_end.train_model(batch_sz,eps,val_splt,model,input_list,output_list,exp_no)





