#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.layers import Dense,Conv2D,Flatten,Input,concatenate,Reshape,Dropout,Activation,BatchNormalization,MaxPooling2D,UpSampling2D
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Model
import tensorflow as tf
import keras
import numpy as np


# In[4]:


def create_model(num_input_channels,input_shapes,num_LSTM_layers,num_LSTM_units,num_dense_layers,num_dense_units,lr,reg_param,inp_optim):
    
    outputs=[];
    inputs=[]
    
    #LSTM Layers Construction
    #-----------------------------------------------------
    for i_channels in range(num_input_channels):
        name_string="input"+str(i_channels+1)        
        inputs.append(Input(shape=(input_shapes[i_channels][1],1),name=name_string))
        for i_individual in range(num_LSTM_layers[i_channels]):
            
            if num_LSTM_layers[i_channels]==1:
                outputs.append(LSTM(num_LSTM_units[i_channels][i_individual], input_shape=(input_shapes[i_channels][1],1), activation='relu')(inputs[i_channels]))
            else:
                if i_individual==0:
                    outputs.append(LSTM(num_LSTM_units[i_channels][i_individual], input_shape=(input_shapes[i_channels][1],1), activation='relu', return_sequences=True)(inputs[i_channels]))
                elif i_individual==num_LSTM_layers[i_channels]-1:
                    outputs[i_channels]=LSTM(num_LSTM_units[i_channels][i_individual], input_shape=outputs[i_channels].shape, activation='relu')(outputs[i_channels])
            
            
                else:
                    outputs[i_channels]=LSTM(num_LSTM_units[i_channels][i_individual], input_shape=outputs[i_channels].shape, activation='relu', return_sequences=True)(outputs[i_channels])
                
    #-------------------------------------------------------
    print("LSTM Layers Constructed")
    
    # Concatenate the outputs of all LSTM layers
    concat=[]
    #print(outputs[0].get_shape().as_list())
    #print(outputs[1].get_shape().as_list())
    #print(outputs[2].get_shape().as_list())
    #print(outputs[3].get_shape().as_list())
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
    input1=Input(shape=(2,1)) # inputs
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
    
    inputs.append(input1)
    
    model=Model(inputs=inputs,outputs=[output])
    
    optim=Adam(lr=lr,clipnorm=1.,amsgrad=True)
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



    



