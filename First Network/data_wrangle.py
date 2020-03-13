#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import re
from copy import deepcopy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textblob import TextBlob 
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

def isNaN(num):
    return num != num    

def remove_stopwords(example_sent):
  
#example_sent = "This is a sample sentence, showing off the stop words filtration."
  
    stop_words = set(stopwords.words('english')) 

    word_tokens = word_tokenize(example_sent) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 

    return filtered_sentence

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

# Start of script

data=pd.read_csv("Concatenate.csv")

#----------------------------------
input_column_list=[9,10,11,12,16,17,19,22,24,37,40,63,66,70,83,85,87];
input_default=["unknown","normal","normal","unknown","unknown","unknown",0,"unknown","unkown","unknown","unknown","unknown","unknown","unknown","unknown","N","unknown"];

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

#--------------

print("Relevant Columns Selected")

'''
with open('input_data', 'wb') as f:
    pickle.dump(input_data, f)
    
with open('output_data', 'wb') as f:
    pickle.dump(output_data, f)
with open('narrative_data', 'wb') as f:
    pickle.dump(narrative_data, f)
with open('synopsis_data', 'wb') as f:
    pickle.dump(synopsis_data, f)



save_input=pd.DataFrame(input_data);
save_output=pd.DataFrame(output_data);
save_narrative=pd.DataFrame(narrative_data);
save_synopsis=pd.DataFrame(synopsis_data);

with open('save_input', 'wb') as f:
    pickle.dump(save_input, f)
    
with open('save_output', 'wb') as f:
    pickle.dump(save_output, f)
with open('save_narrative', 'wb') as f:
    pickle.dump(save_narrative, f)
with open('save_synopsis', 'wb') as f:
    pickle.dump(save_synopsis, f)
'''
#print(save_input.shape)
#print(save_output.shape)
#print(save_narrative.shape)
#print(save_synopsis.shape)


    #if col=="Environment":
    #    print(col=="Environment")
    
    #print(data[col][:])
    
    
#print(input_data[1][1])
    


# In[2]:
print("Starting Data Cleaning")

num_datapoints=len(input_data[14][:])
num_columns=len(input_data);
input_default=["unknown","normal","normal","unknown","unknown","unknown",0,"unknown","unkown","unknown","unknown","unknown","unknown","unknown","unknown","N","unknown"];

default_datatype=[type(input_default[i]) for i in range(len(input_default))];

print("Cleaning Input Feature Data")

for i_column in range(num_columns):
    if i_column==6:                #Hard coded column number
        continue;
    for i_datapoint in range(num_datapoints):
        #print(input_data[i_column][i_datapoint])
        #if is_integer(input_data[i_column][i_datapoint]):
         #   print("booyah")
         #   input_data[i_column][i_datapoint]=input_default[i_column];
         #   continue;
        
        string_1=input_data[i_column][i_datapoint];
        string_1=string_1.lower()                     #Lowercase

        string_1 = re.sub(r'[^\w\s]','',string_1)     # remove punctuation
        input_data[i_column][i_datapoint]=string_1

# Convert to Pd dataframe         
input_data=pd.DataFrame(input_data);

input_data=input_data.drop(input_data.index[[0,2,11,12,13]],axis=0)

print("Done")
print("SELECTING ONLY 5 RELEVANT COLUMNS")
# For first iteration of the project, select 5 columns


input_data.iloc[:,[30]]

input_data=input_data.iloc[[0,2,4,6,9],:]



# In[3]:
print("Cleaning Narrative data")

# Narrative data pre-processing
narrative_data_main=deepcopy(narrative_data[0])
for i_datapoint in range(num_datapoints):

        
    string_1=narrative_data_main[i_datapoint];
    string_1=string_1.lower()                     #Lowercase

    string_1 = re.sub(r'[^\w\s]','',string_1)     # remove punctuation
    narrative_data_main[i_datapoint]=string_1
    
    narrative_data_main[i_datapoint]=TreebankWordDetokenizer().detokenize(remove_stopwords(narrative_data_main[i_datapoint])) #remove stopwords
print("Done")
print("Obtaining Narrative Sentiment Scores")
narrative_scores=[];
pos_weight=0.9;
sub_weight=0.1;
for i in range(len(narrative_data_main)):

    pos_score=TextBlob(narrative_data_main[i]).sentiment[0]
    sub_score=TextBlob(narrative_data_main[i]).sentiment[1]
    narrative_scores.append(pos_weight*pos_score+sub_weight*sub_score)
print("Done")

print(len(narrative_scores))

with open('risk_mapping.pkl', 'rb') as f:
    risk_data = pickle.load(f)

input_data=input_data.values.tolist()

remove_counter=0
i=0
while i<len(risk_data):

    if risk_data[i]==3:

        for j in range(len(input_data)):
            input_data[j].pop(i)

        risk_data.pop(i)
        narrative_scores.pop(i)
        remove_counter=remove_counter+1
        continue;
    i=i+1
    if remove_counter > 6000:
        break;

class_size=np.zeros([5])

for i in range(len(risk_data)):

    class_size[risk_data[i]-1]+=1;

print(class_size)





#--------------------------- One-hot encode risk-mapping

encoded_risk=np.zeros([len(risk_data),5])
for i in range(len(risk_data)):

    for i_enc in range(5):

        if int(i_enc+1)==risk_data[i]:
            #print(i,i_enc)
            encoded_risk[i,i_enc]=1
            break;
print(risk_data[0])
print(encoded_risk[0])

#--------------------------

#

print("Save narrative and input data using pickle, and encoded risk matrix")
#save the required data

input_data,narrative_scores,encoded_risk=unison_shuffled_copies(input_data,narrative_scores,encoded_risk)

with open('input_data_test', 'wb') as f:
    pickle.dump(input_data, f)
 
with open('narrative_data_test', 'wb') as f:
    pickle.dump(narrative_scores, f)
with open('risk_encoding', 'wb') as f:
    pickle.dump(encoded_risk, f)

# In[ ]:




