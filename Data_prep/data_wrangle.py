#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import math
import pandas as pd
from copy import deepcopy

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def isNaN(num):
    return num != num    
    
data=pd.read_csv("Concatenate.csv")
print(data.shape)

#----------------------------------
input_column_list=[9,10,11,12,16,17,19,22,24,37,40,63,66,70,83,85,87];
input_default=["unknown","normal","normal","unkown","unknown","unknown",0,"unknown","unkown","unknown","unknown","unknown","unknown","unknown","unknown","no","unknown"];

output_column_list=[88,89];

narrative_column_list=[91,92,93];
synopsis_column_list=[95];
#-----------------------------------
count_col=0;

input_data=[];
output_data=[];
narrative_data=[];
synopsis_data=[];
#----------------------------------Create the lists, no data wrangled
for col in data.columns:
    
    if count_col in input_column_list:
        
        input_data.append(list(data[1:][col]));
           
    if count_col in output_column_list:
        
        output_data.append(list(data[1:][col]));
    
    if count_col in narrative_column_list:
        
        narrative_data.append(list(data[1:][col]));
    
    if count_col in synopsis_column_list:
        
        synopsis_data.append(list(data[1:][col]));
    
    count_col=count_col+1;
    
#----------------------------------
#Remove NaN values from input
if len(input_data)!=len(input_default):
    raise Exception("Correct data");

for i_list in range(len(input_data)):
    
    sublist=deepcopy(input_data[i_list][:]);
    
    for j_list in range(len(sublist)):
        
        if isNaN(sublist[j_list]):
            input_data[i_list][j_list]=input_default[i_list];
#------------------------------------
#Remove records with NaN in output(Result space)

index_toremove=[];

for i_list in [0]: # Get indices to remove
    
    sublist=deepcopy(output_data[i_list]);
    
    for j_list in range(len(sublist)):
            
        if isNaN(sublist[j_list]):
            
            index_toremove.append(j_list);
print(np.array(index_toremove).shape)                        
#Remove the records from all lists correspnding to index_toremove

for i_remove in range(len(index_toremove)):
    
    index_current=index_toremove[len(index_toremove)-i_remove-1];

    for i_input in range(len(input_data)):
        del(input_data[i_input][index_current]);
    for i_output in range(len(output_data)):
        del(output_data[i_output][index_current]);
    for i_narrative in range(len(narrative_data)):
        del(narrative_data[i_narrative][index_current]);
    for i_synopsis in range(len(synopsis_data)):
        del(synopsis_data[i_synopsis][index_current]);

#--------------------------------------------------------

# Sanity Check: No  NaNs anywhere:

for i in range(len(input_data)):
    
    sublist=deepcopy(input_data[i]);
    for j in range(len(sublist)):
        
        if isNaN(input_data[i][j]):
            raise Exception("NaN found");
            


save_input=pd.DataFrame(input_data);
save_output=pd.DataFrame(output_data);
save_narrative=pd.DataFrame(narrative_data);
save_synopsis=pd.DataFrame(synopsis_data);
print(save_input.shape)
print(save_output.shape)
print(save_narrative.shape)
print(save_synopsis.shape)
    #if col=="Environment":
    #    print(col=="Environment")
    
    #print(data[col][:])
    
    
#print(input_data[1][1])
    


# In[ ]:




