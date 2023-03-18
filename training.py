import torch
import torch.nn as nn
import random
import reader

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        #conv NN
        self.conv=torch.nn.Sequential(
            #beginning with 6*8*8 input
            torch.nn.Conv2d(6,4,2), #6-channel 8*8 input--2*2 kernel-->7*7 output. using 4-channel. 
            torch.nn.ReLU(),
            torch.nn.Conv2d(4,1,4), #4-channel 7*7 input-->4*4 kernel->4*4 output. 1-channel output.
            #dilation seems reasonable. consider adding layer
            torch.nn.ReLU(),
            #the last layer should generate 4-channel 4*4 output
            torch.nn.AvgPool2d(2) #2*2 pooling kernel. just a random guess choice.
            #now should have 4-channel 3*3 output
        )

        #linear layer to generate a single output
        self.lin=torch.nn.Sequential(
            torch.nn.Linear(4,2)
        )
    def forward(self,x):
        '''
        x: 6*8*8
        '''
        #print(x)
        x = self.conv(x)
        #print(x)
        x = torch.flatten(x,1)
        #print(x)
        x = self.lin(x)
        #print(x)
        return x

def train_set(set,model,loss_fn,optimizer):
    '''
    trains model over a given set with specified loss_fn and optimizer.
    '''
    for case in set:
        datum = torch.tensor(case[0],dtype = torch.float32) #everything related to board; list of list of lists of arrays, 7*8*8 in total
        ending = torch.tensor([float(case[1][0]),float(case[1][1])])#the ending of the game in which this board showed up
        #print(datum)
        prediction = model(datum)
        prediction = torch.tensor([float(prediction[0][0]),float(prediction[0][1])])
        #print(ending, prediction)
        loss = loss_fn(prediction,ending)
        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

def train_loop(sets,epochs=10):
    model=NeuralNet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=10)
    print("training...")
    for epoch in range(epochs):
        test_set_number = random.randint(0,9)
        print(" training loop:",epoch)
        print(" testing set is: set",test_set_number)
        train_sets(sets,test_set_number,model,loss_fn,optimizer)
        print(" training finished, now testing...")
        test_set(sets,test_set_number,model,loss_fn)
    return model, loss_fn, optimizer

def train_sets(sets,test_set_number,model,loss_fn,optimizer):
    for i in range(len(sets)):
        if i == test_set_number:
            print("  bypass testing set, index =", i)
            continue
        print("  training on set", i)
        train_set(sets[i],model,loss_fn,optimizer)

def test_set(sets,test_set_number,model,loss_fn):
    print("  testing on testing set, index =", test_set_number)
    test_loss = 0
    for case in sets[test_set_number]:
        with torch.no_grad():
            datum = torch.tensor(case[0],dtype = torch.float32)
            ending = torch.tensor([float(case[1][0]),float(case[1][1])])
            prediction = model(datum)
            prediction = torch.tensor([float(prediction[0][0]),float(prediction[0][1])])
            loss=loss_fn(prediction,ending)
            test_loss+=loss
    test_loss = test_loss / len(sets[test_set_number])
    print("  average test loss on set:",test_loss)
    

def main():
    #white,black,sets = reader.readtxt()
    #model, loss_fn, optimizer = train(sets,0)
    return None
main()


        

