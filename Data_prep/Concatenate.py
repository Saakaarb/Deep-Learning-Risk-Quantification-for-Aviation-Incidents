import os
#os.chdir('C:/Users/tragu/Contacts/Documents/Nicolas/USA/Stanford/CS320 Deep Learning/Projet/Database/All') #Specify files location, all in one folder
os.chdir('/home/saakaar/Desktop/Q2/CS230/Project_data/Database.')
#Filenames=os.listdir('C:/Users/tragu/Contacts/Documents/Nicolas/USA/Stanford/CS320 Deep Learning/Projet/Database/All') #Get the filenames
Filenames=os.listdir('/home/saakaar/Desktop/Q2/CS230/Project_data/Database.')
File = open('Concatenate.csv', 'w') #Create the merged file

for i in range(len(Filenames)):
    Currentfile=open(Filenames[i], 'r')
    Currentfile.readline(), Currentfile.readline(), Currentfile.readline()#Removes the heading
    File.write(Currentfile.read()) #Append current file
    Currentfile.close()
File.close()


