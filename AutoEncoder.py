# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:40:44 2020

@author: Ankush
"""

#AutoEncoders

#Importing the Libraries
import numpy as np #because we need to work with arrays
import pandas as pd
#importing torch libraries
import torch
import torch.nn as nn #torch module to build neural network
import torch.nn.parallel #for parallel computation
import torch.optim as optim #for optimization
import torch.utils.data #some torch tools
from torch.autograd import Variable #for stocastic gradient descent

#Importing the dataset
movies= pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python',encoding='latin-1')
# 'encoding = 'latin-1'' is used to read special characters in the dataset which is not possible with typical encoding = utf8
#'header =none' means no special row to mention he headings 
users= pd.read_csv('ml-1m/users.dat',sep='::',header=None, engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None, engine='python',encoding='latin-1')

#Preparing th training set and the test set
training_set= pd.read_csv('ml-100k/u1.base',delimiter='/t')#delimiter refers the seperation key which is 'tab' here 
training_set= np.array(training_set, dtype='int')
test_set= pd.read_csv('ml-100k/u1.test',delimiter='/t')
test_set= np.array(test_set, dtype='int')

nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))#finding the maximum number of users by finding the maximum index value which could be in either training or test set
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))#finding the maximum number of movies by finding the maximum index value  which could be in either training or test set

#converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data=[] # a function to create the list of list
    for id_users in range(1, nb_users+1):#loop to store movie ratings of all users
        id_movies= data[:,1][data[:,0]== id_users]#an array will be created containing the indexes of movies rated by respective user
#ex.     user 1 - [3,2,4,1,5,3,4,5,2,3,4,2]               "[data[:,0]==id_users]" means we are taking the movie_ids of only those movies rated by one user only, denoted by 'id_users'.Thats the way to write it.cont.
#        user 2 - [1,3,3,4,5,2,5,2,3,1,4,3,4]             Remember it is in LOOP. So we will place the data of each user(movies that has been rated) using loop.
        id_ratings= data[:,2][data[:,0]== id_users]#an array will be created containing the indexes of ratings rated by respective user
        ratings= np.zeros(nb_movies)#creating a huge array of zeros with total columns equal to total no. of movies. 
        ratings[id_movies - 1]= id_ratings #each value in id_ratings will get stored in the "ratings" array at respective indexes
        new_data.append(list(ratings))#the ratings list (which is array) will be placed in another list(list of list) and the process will be iterated for all users
    return new_data    
training_set=convert(training_set)#sending the training set as a variable to convert() fn
test_set=convert(test_set) #sending the test set as variable to convert() fn
                                                               
#Creating the architechture of the Neural Network
#we are creating a class in which we will define the architechture of our Autoencoder
class SAE(nn.Module): #Module is the class of 'nn' that contains various materials(basically different functions) to build the AE.It is the parent class of SAE class
    def __init__(self, ): #it is the fn of class SAE with 'self' as class's default object. The gap after 'self' indicates that we are importing the modules of inherited class(which is 'Module' class). 'self' basically represents AE.
        super(SAE,self).__init__ #super is a pre-defined fn. It helps to import the functions or modules of inherited class.Its format includes(here) mentioning the class name, class's object as parameters and attaching it to the fn in which it is going to be used.
        self.fc1= nn.Linear(nb_movies, 20) #now we are creating the connection layer.'fc1' simply represents 'full connection' which is an object of 'self'. We are importing the 'Linear' class of 'nn' module.It has 2 parameters , 1st one represents the total no.of input nodes.cont 
        #which ,here, is total no. of movies. The 2nd parameter is total no. of hidden nodes in the 1st layer(it is an experimented value , can be optimized). Similarly, we will create other input layers.
        self.fc2= nn.Linear(20, 10) #this is the 2nd hidden layer with 1st parameter as HNs of 1st layer and 2nd parameter as HNs of 2nd layer.(Thats the format) 
        self.fc3= nn.Linear(10, 20) #this is the 3nd hidden layer with 1st parameter as HNs of 2nd layer and 2nd parameter as HNs of 3rd layer. 
        self.fc4= nn.Linear(20, nb_movies)#this is the 4nd hidden layer with 1st parameter as HNs of 3rd layer and 2nd parameter as HNs of 4th layer(last layer).The last hidden layer has as many nodes as the input layer()
        self.activation= nn.sigmoid() #using 'sigmoid' activation fn of 'nn' module to activate the nodes in Hidden layers.
    def forward(self, x): #it is the fn to encode and decode the layers of AE along with applying activation fns to their nodes.here x is the input nodes vector(contains all the movies rated by current user).In the end 'x' will be modified into output layer which will be our result.  
        x = self.activation(self.fc1(x))#We are encoding our input layer vector into shorter hidden layer vector.Full connections means having weights between nodes.  This line will return the first encoded vector(which is modified 'x') which will then form our first hidden layer.cont.
        #It will be continued for all the layers until we get the output layer .This is the format.
        x = self.activation(self.fc2(x))#The 1st encoded vector will be encoded into another shorter encoded vector which is going to be 2nd encoded vector .
        x = self.activation(self.fc3(x))#The 2nd encoded vector will be encoded into another shorter encoded vector which is going to be 3rd encoded vector .
        x = self.fc4(x)#It is the output layer.We don't need to add the activation here since the layer will use the default linear activation.
        return x #it is going to return the modified value of 'x' which is the output layer vector 
sae = SAE()#creating an object of the class
criterion = nn.MSELoss()#Define a criterion for the loss function. We'll use it at the training phase. The loss will be the Mean Squared Error.
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)#calling 'RMSprop' class of 'optim' module of torch.We are gonna import all the parameters of 'sae' class using 'parameters' object.ip-The SAE class inherits methods from the nn.Module class and the 'parameters' is from the nn.Module class.cont.
#(in case you are wondering from where 'parameters' is imported).'lr' is the learning rate and 'weight_decay' slows down the learning rate.(after a lot of epochs, we have to slow down the jump to reach the minima of the gradient).both 'lr' and 'weight_decay' are tunable.
#note-(learning rate) Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient OR In machine learning and statistics, the learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving towards a minimum of a loss function.

#Training the SAE
nb_epochs=200#no. of epochs decided by us.
for epochs in range(1,nb_users+1):# using the loop to run the code till all the epochs
    train_loss=0 #the variable that stores the loss value(difference between the original and new values)for each user(and gets reset for the next user)
    s=0.#used to calculate RME(Root Mean Square error)
    for id_users in range(nb_users):#this loop is to apply the Autoencoder(sae) to all users one by one
        input=Variable(training_set[id_users]).unsqueeze(0)#torch dosent work on simple 1 dimension arrays. we need to create a batch or multidimentional array.It was done through 'squeeze' fn of 'Variable' class
        target=input.clone()#we are making the clone of our input vector which is represented by 'input' variable.
        if torch.sum(target.data > 0) > 0:#checking if the sum of all the values(ratings) in the 'target' variable is greater than 0. "(target.data> 0)" will return the indexes of all values of target greater than zero and then their summation is checked(through 'torch.sum') whether the summation is > 0.
            output= sae(input)#executing the autoencoder with 'input' as argument.So, it goes like this. We call sae -> it calls nn.Module -> it accesses the __call__ -> it executes the forward() which it finds within our SAE class.The forward function should be overridden by all subclasses.
            #Ques-Shouldn't it be output = sae.forward(input) instead of sae(input) ans-It is the same since forward is the only other function in sae other than the init fn.also it is the only fn with a parameter which here is 'x'.
            target.require_grad= False# We can just calculate the gradient w.r.t the input instead of doing it twice for the input and the target hence reducing computations. This is only true for autoencoders.
            output[target==0] = 0# Make sure we only deal with the movies the user rate for the predictions variable. The movies that the user didn't rate will not count for the computations of the error.'[target==0]' returns the indexes of all the values of target equal to 0. 
            #Now, all the values of output with the same indexes as returned by the inside code will be made equal to zero. They simply wont take part in loss calculation and optimization.
            loss=criterion(output,target)#calculating the loss(this is not the actual loss,just a part of it) by calling criterion fn with arguments output(predicted values) and target(initial values)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0)+ 1e-10)#The average of the error, but only considering the movies that are rated by the user.We add 1e-10 to make sure that the denominator is not 0.It is just a parameter needed to calculate the loss.
            loss.backward()#backward is the pre-defined function of nn that updates weights on the back-propagation.Its not the direction of weight update, but the direction of update pass. It is the back in back-propagation.
            train_loss+=np.sqrt(loss.item()*mean_corrector)# Computing the relevant mean of loss using the mean_corrector variable.this is the formula.
            s+=1.#Increasing the numbers of users who rated at least one movie.it will be used as final parameter to calculate the loss.
            optimizer.step()# This function decides the amount by which the weights will be updated.Basically, it decides the intensity of weight updates. We are using 'step()' which is an object of 'RMSprop' class to implement the 'optimizer' fn.(we didn't use such module previously)
    print('epoch: '+str(epochs)+'loss :'+str(train_loss/s) ) #printing the epoch number and the loss value for that whole epoch. Remember, it is printed outside 'if' condition . We divide the train_loss by 's' because apparently,it adds the losses of all the epochs.(ex-- (0.54+0.67+0.48) / 3)   
# how the optimizer.step() call makes use of the "loss" object? When loss.backward() is called, gradients are being computed for the weights of the network. These computed gradients are used later on by the optimizer to update the weights.

#Testing the SAE
test_loss=0
s=0.
for id_users in range(nb_users):#we dont need "loss.backward()" and "optimizer.step()" in the testing phase.
      input=Variable(training_set[id_users]).unsqueeze(0)
      target=Variable(test_set[id_users]).unsqueeze(0)
      if torch.sum(target.data > 0) > 0:
          output=sae(input)
          target.required_grad=False
          output[target==0] = 0
          loss=criterion(output,target)
          mean_corrector= nb_movies/float(torch.sum(target.data > 0) + 1e-10)
          test_loss+=np.sqrt(loss.item()*mean_corrector)#it will accumulate all the losses of all the users and then in the line 113,we will divide the test_loss by s , to get the actual test loss value.
          s+=1.    
      print('test loss: '+str(test_loss/s))
#1.how do we store the trained model and how to use it ? Otherwise we have to train the mode again and again.
#2."target" and "input" should be same because we will take the "input" as input and the "output" generated will be compared to "input" or "target". "target" basically indicates the original input values, otherwise "input" variable is enough . 
#Now we are using training set input for the test set i.e." input = Variable(training_set[id_user]).unsqueeze(0)" but our "target" variable(which is the variable denoted for the original values) is taken  from the test dataset(u1.test) which has different movies that were rated by the users.(different movie_ids basically).
#So if "output" and "target" are different, dont you think our testing code is wrongly build?? 

#finding the movie ratings of a particular movie for a particular user.note- it wont give a direct loss value but a value close to 3. If the value is greater than 3, that means the review is positive else negative.cont.  
target_user_id = 3 #we can also calculate the actual loss by using same codes in training or test set but we chose a simpler way with few lines of codes. 
target_movie_id = 327
input = Variable(training_set[target_user_id-1]).unsqueeze(0)
output = sae(input)
output_numpy = output.data.numpy()
print (''+ str(output_numpy[0,target_movie_id-1]))
#I am interpreting it as if the user watches the movie with target_movie_id and if result obtained from code is above than 3 that means user will like the movie and if less than 3 than user will not like the movie.

















