#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.layers import Dense,Conv2D,Flatten,Input,concatenate,Reshape,Dropout,Activation,BatchNormalization,Embedding,Lambda
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Model
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import keras
import numpy as np


# In[4]:


def create_model(num_input_channels,vocab_sizes,output_size_embedding,output_size_narrative_embedding,input_shapes,num_LSTM_layers,num_LSTM_units,num_dense_layers,num_dense_units,learning_rate,reg_param,inp_optim,embedding_matrix):
   

    outputs=[]
    inputs=[]
    squeezed=[]   
    
    # Input Layer Constrution: Categorical text Inputs:
    
    for i_channels in range(num_input_channels):
        name_string="input"+str(i_channels+1)
        inputs.append(Input(shape=(input_shapes[i_channels][1],1),name=name_string))
        print(inputs[i_channels].shape)
   
    # Squeeze to enable input to embedding layer:

    for i_channels in range(num_input_channels):
        squeezed.append(Lambda(lambda x:keras.backend.squeeze(x,axis=-1))(inputs[i_channels]))
        print(squeezed[i_channels].shape)
    # Embedding Layers Construction: Categorical text inputs
    for i_channels in range(num_input_channels):
        if i_channels==num_input_channels-1:
            outputs.append(Embedding(input_dim=vocab_sizes[i_channels],output_dim=output_size_narrative_embedding,input_length=input_shapes[i_channels][1],weights=[embedding_matrix],trainable=False)(squeezed[i_channels]))
        else:
            outputs.append(Embedding(input_dim=vocab_sizes[i_channels],output_dim=output_size_embedding,input_length=input_shapes[i_channels][1])(squeezed[i_channels]))

    #LSTM Layers Construction: Categorical text inputs
    #-----------------------------------------------------
    for i_channels in range(num_input_channels):
        
        for i_individual in range(num_LSTM_layers[i_channels]):

            if num_LSTM_layers[i_channels]==1:
                outputs[i_channels]=LSTM(num_LSTM_units[i_channels][i_individual], activation='relu')(outputs[i_channels])
            else:
                if i_individual==0:
                    outputs[i_channels]=LSTM(num_LSTM_units[i_channels][i_individual], activation='relu', return_sequences=True)(outputs[i_channels])
                elif i_individual==num_LSTM_layers[i_channels]-1:
                    outputs[i_channels]=LSTM(num_LSTM_units[i_channels][i_individual], activation='relu')(outputs[i_channels])
        
        
                else:
                    outputs[i_channels]=LSTM(num_LSTM_units[i_channels][i_individual], activation='relu', return_sequences=True)(outputs[i_channels])
    

    


    #-------------------------------------------------------
    print("LSTM Layers Constructed")
    
    # Concatenate the outputs of all LSTM layers
    concat=[]
    for i_concat in range(num_input_channels-1):
        if i_concat==0:
            output=concatenate([outputs[0],outputs[1]])
    #        print(output.get_shape().as_list())
        else:
            output=concatenate([output,outputs[i_concat+1]])
    #Dense Layers Construction
    #-------------------------------------------------------
    
    print(output.get_shape().as_list()) 
    print("Create Dense Layers")
    
    #-----------------------------
    #Dense Layer Creation
    output=Reshape((-1,1))(output)
    print(output.get_shape().as_list())

    input1=Input(shape=(1,1)) # inputs
     # 
    output=concatenate([output,input1],axis=1) #add inputs to LSTM outputs
    output=Flatten()(output)

    for i_dense in range(num_dense_layers):
        if i_dense==num_dense_layers-1:
            output=Dense(num_dense_units[i_dense],activation='softmax',kernel_regularizer=regularizers.l2(reg_param))(output)
        else:
            output=Dense(num_dense_units[i_dense],activation='relu',kernel_regularizer=regularizers.l2(reg_param))(output)
            output=BatchNormalization()(output)
    #----------------------------
    print("Done")    
    inputs.append(input1)
    
    model=Model(inputs=inputs,outputs=output)
    
    optim=Adam(lr=learning_rate,clipnorm=1.,amsgrad=True)
    model.compile(optimizer=optim,loss="categorical_crossentropy",metrics=["accuracy"])

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
    	json_file.write(model_json)

    model.summary()

    return model

def train_model(batch_sz,eps,val_splt,model,input_list,output_list,exp_no):

    csv_logger=CSVLogger('training_%d.csv'%(exp_no));

    checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(input_list,output_list,batch_size=batch_sz,epochs=eps,validation_split=val_splt,callbacks=[csv_logger,checkpoint])
    
    model.save('Network_Expt_%d.h5'%(exp_no))



    



