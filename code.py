#importing natural language toolkit
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
stemmer = LancasterStemmer()

#training data set
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})
#print ("%s sentences in training data" % len(training_data))
#print(training_data)

#oraganizing data
words = []
classes = []
documents = []
ignore_words = ['?']
#looping through each sentance in our training data
for pattern in training_data:
	#tokenize each word in the sentance
	w = nltk.word_tokenize(pattern['sentence'])
	#adding to word list
	words.extend(w)
	#add to documents in our corpus
	documents.append((w,pattern['class']))
	#add to our classes list
	if pattern['class'] not in classes:
		classes.append(pattern['class'])

#stemming and removing duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
#print(words)

classes = list(set(classes))
#print(classes)

#create training data
training = []
output = []
#empty array for output
output_empty = [0]*len(classes)
#print(list(output_empty))

#training set, bag of words for each sentence
for doc in documents:
	#initialize our bag of words
	bag = []
	#list of tokenized words for the pattern
	pattern_words = doc[0]
	#stem each word
	pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
	#create out bag of words array
	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)
	training.append(bag)
	#output will be 0 for each tag and 1 for current tag
	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1
	output.append(output_row)
	
#sample training/output
'''i = 0
for i in range(len(documents)):
	w = documents[i][0]
	print([stemmer.stem(word.lower()) for word in w])
	print(training[i])
	print(output[i])
	print("\n")'''
	
import numpy as np
import time

#computing sigmoid non-liniarity
def sigmoid(x):
	z = 1/(1+np.exp(-x))
	return z
	
#converting output of sigmoid function to its derivative
def sigmoid_derivative(x):
	return x*(1-x)
	
def clean_up_sentence(sentence):
	#tokenize the pattern
	sentance_words = nltk.word_tokenize(sentence)
	#stem each word
	sentence_words = [stemmer.stem(word.lower()) for word in sentance_words]
	return sentance_words
	
#return bag of words
def bag_of_words(sentence,words, show_details=False):
	#tokenize pattern
	sentence_words = clean_up_sentence(sentence)
	#bag of words
	bag = [0]*len(words)
	for s in sentence_words:
		for i,w in enumerate(words):
			if w == s:
				bag[i] = 1
				if show_details:
					print("found in bag: %s" %w)
	
	return(np.array(bag))
	
def think(sentence, show_details=False):
	x = bag_of_words(sentence.lower(),words,show_details)
	if show_details:
		print("sentence:",sentence, "\nbag of words: ",x)
	#input layer is our bag of words
	layer_0 = x
	#matrix multiplication of input layer and hidden layers
	layer_1 = sigmoid(np.dot(layer_0, synapse_0))
	#output layer
	layer_2 = sigmoid(np.dot(layer_1, synapse_1))
	return layer_2
	
#neural network training
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
	#print("Traing with %s neurons, alpha:%s, droupout:%s %s" %(hidden_neurons, str(alpha), dropout, dropout, dropout_percent if dropout else ''))
	print("Input matirx: %sx%s Output matrix: %sx%s"%(len(X),len(X[0]),1,len(classes)))
	np.random.seed(1)
	
	last_mean_error = 1
	#randomly initialize our weights with mean 0
	synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons))-1
	synapse_1 = 2*np.random.random((hidden_neurons, len(classes)))-1
	
	previous_synapse_0_weight_update = np.zeros_like(synapse_0)
	previous_synapse_1_weight_update = np.zeros_like(synapse_1)
	
	synapse_0_direction_count = np.zeros_like(synapse_0)
	synapse_1_direction_count = np.zeros_like(synapse_1)
	
	for j in iter(range(epochs+1)):
		#feed forword through layers 0, 1, 2
		layer_0 = X
		layer_1 = sigmoid(np.dot(layer_0, synapse_0))
		
		#if(dropout):
		#	layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))], 1-dropout_percent)[0] * (1.0/(1-dropout_percent))
	
		layer_2 = sigmoid(np.dot(layer_1, synapse_1))
		
		#error
		layer_2_error = y - layer_2
		
		if (j%10000)==0 and j>5000 :
			#break if the 10k itretion's error is more than last iteration
			if (np.mean(np.abs(layer_2_error))) < last_mean_error :
				#print(np.mean(np.abs(layer_2_error)), " ", last_mean_error)
				last_mean_error = np.mean(np.abs(layer_2))
			else :
				#print("break: ", np.mean(np.abs(layer_2_error)), " ", last_mean_error)
				break
		
		#ackpropagation
		layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
	
		#layer_1 error contribution
		layer_1_error = layer_2_delta.dot(synapse_1.T)
		#print("\n", synapse_1.T, "\n")
		
		#backpropagation
		layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
		
		synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
		synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
		
		'''if(j > 0):
			synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
			synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))'''        
		
		synapse_1 += alpha * synapse_1_weight_update
		synapse_0 += alpha * synapse_0_weight_update
        
		#prev_synapse_0_weight_update = synapse_0_weight_update
		#prev_synapse_1_weight_update = synapse_1_weight_update
		
	now = datetime.datetime.now()
	
	#persistant synapse_0
	synapse = {'synapse0': synapse_0.tolist(), 
			   'synapse1': synapse_1.tolist(),
			   'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
	synapse_file = "synapses.json"

	with open(synapse_file, 'w') as outfile:
		json.dump(synapse, outfile, indent=4, sort_keys=True)
		

X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")

# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
	results = think(sentence, show_details)
	results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ]
	results.sort(key=lambda x: x[1], reverse=True) 
	return_results =[[classes[r[0]],r[1]] for r in results]
	print ("%s \n classification: %s" % (sentence, return_results))
	return return_results

classify("sudo make me a sandwich")
classify("how are you today?")
classify("talk to you tomorrow")
classify("who are you?")
classify("make me some lunch")
classify("how was your lunch today?")
print()
classify("good day", show_details=True)