import os
import pickle
import numpy as np


os.chdir('C:/Users/tragu/Contacts/Documents/Nicolas/USA/Stanford/CS320 Deep Learning/Projet') #Set directory

with open('output_data', 'rb') as f: #Importing data
    output_data = pickle.load(f)

def risk_mapping(output_data):
    risk=[0]*len(output_data[0]) #Create output list

    #Define the lists of risks as can be found in the reports. Punctuation may vary
    high_risk=['General Declared Emergency', 'General Physical Injury / Incapacitation', 'Flight Crew Inflight Shutdown', 'Air Traffic Control Separated Traffic', 'Aircraft Damaged', 'Aircraft Aircraft Damaged' ]

    moderately_high_risk=['General Evacuated', 'Flight Crew Regained Aircraft Control', 'Air Traffic Control Issued Advisory / Alert', 'Flight Crew Landed in Emergency Condition', 'Flight Crew Landed In Emergency Condition']

    medium_risk=['General Work Refused', 'Flight Crew Became Reoriented', 'Flight Crew Diverted', 'Flight Crew Executed Go Around / Missed Approach', 'Flight Crew Overcame Equipment Problem', 'Flight Crew Rejected Takeoff', 'Flight Crew Took Evasive Action', 'Air Traffic Control Issued New Clearance']

    moderately_medium_risk=['General Maintenance Action', 'General Flight Cancelled / Delayed', 'General Release Refused / Aircraft Not Accepted', 'Flight Crew Overrode Automation', 'Flight Crew FLC Overrode Automation', 'Flight Crew Exited Penetrated Airspace', 'Flight Crew Requested ATC Assistance / Clarification', 'Flight Crew Landed As Precaution', 'Flight Crew Returned To Clearance', 'Flight Crew Returned To Departure Airport', 'Aircraft Automation Overrode Flight Crew']

    low_risk=['General Police / Security Involved', 'Flight Crew Returned To Gate', 'Aircraft Equipment Problem Dissipated', 'Air Traffic Control Provided Assistance', 'General None Reported / Taken', 'Flight Crew FLC complied w / Automation / Advisory']

    #Map
    k=0
    for i in output_data[0]:
        for j in i.split(';'): #One event can lead to multiple outcomes
            j=j.strip()
            if j in low_risk and risk[k]<=1:
                risk[k]=1
            elif j in moderately_medium_risk and risk[k]<=2:
                risk[k]=2
            elif j in medium_risk and risk[k]<=3:
                risk[k]=3
            elif j in moderately_high_risk and risk[k]<=4:
                risk[k]=4
            elif j in high_risk and risk[k]<=5:
                risk[k]=5
        k+=1
    return risk


f = open("risk_mapping.pkl","wb") #Exporting data
pickle.dump(risk_mapping(output_data), f)
f.close()






