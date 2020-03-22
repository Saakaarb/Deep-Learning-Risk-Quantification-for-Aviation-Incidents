# Deep-Learning-Risk-Quantification-for-Aviation-Incidents
Student Repository for CS230 at Stanford University

This project was done in collaboration with Nicolas Tragus, MS Candidate, Mechanical Engineering, Stanford University  


In a crisis, pilots have to decide between courses of actions that can potentially endanger lives versus courses that can cause enormous monetary losses, e.g. by diverting to the nearest airport. In light of this, the current work aims at developing a too that can quickly quantify the risk of a situation in the air, and can guide pilots to take actions in light of historical data, i.e. what pilots did before similar situations, and based on previous outcomes, perhaps suggest a best decision. 
We use a combination of LSTM and fully connected layers to analyze the information given as categorical strings (almost all data), and a sentiment analysis or word embedding for the narrative written by the pilots. Then, a combination of dense layers with a softmax activation outputs the probability for every risk class given the current situation. 
