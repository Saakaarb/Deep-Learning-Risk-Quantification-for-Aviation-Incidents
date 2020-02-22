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



with open('input_data_test', 'rb') as f:
    input_data = pickle.load(f)
    
with open('narrative_data_test', 'rb') as f:
    narrative_scores = pickle.load(f)    

with open('risk_encoding', 'rb') as f:
    encoded_risk = pickle.load(f)
    
#Convert Pd to list

input_data=input_data.values.tolist()

num_input_columns=len(input_data)

col=[];

for i_inp in range(num_input_columns):
    if i_inp==2:
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

# Fit tokenizers

for i_token in range(num_string_inps):

    tokenizers[i_token].fit_on_texts(col[i_token])


#-------------------------------
# One hot encoding
encoded_col=[];
for i_token in range(num_string_inps):
    encoded_col.append(np.expand_dims(np.array(tokenizers[i_token].texts_to_matrix(col[i_token])),axis=2))
    print(encoded_col[i_token].shape)

#--------------------------------:


integer_inps=np.array(integer_inps)
narrative_scores=np.array(narrative_scores)
integer_inps=np.expand_dims(integer_inps,axis=1)
narrative_scores=np.expand_dims(narrative_scores,axis=1)
numerical_inputs=np.concatenate((integer_inps,narrative_scores),axis=1)


encoded_col.append(np.expand_dims(numerical_inputs,axis=2))
print(encoded_col[4].shape)

input_list=deepcopy(encoded_col)
output_list=deepcopy([encoded_risk])
#-----------------------------

# Create the model

exp_no=1;


num_input_channels=num_string_inps
num_LSTM_layers=[2,2,2,2];
num_LSTM_units=[[16,16],[16,16],[16,16],[16,16]];
num_dense_layers=2;
num_dense_units=[10,5]; # Last dense units always has to be number of risk categories (5)
learning_rate=10**(-2);
inp_optim="Adam";
reg_param=0;
batch_sz=256;
eps=10;
val_splt=0;



input_shapes=[]
for i_inp in range(num_input_channels):

    input_shapes.append(encoded_col[i_inp].shape)
#print(input_shapes[1][1])

model=back_end.create_model(num_input_channels,input_shapes,num_LSTM_layers,num_LSTM_units,num_dense_layers,num_dense_units,learning_rate,reg_param,inp_optim)

back_end.train_model(batch_sz,eps,val_splt,model,input_list,output_list,exp_no)





