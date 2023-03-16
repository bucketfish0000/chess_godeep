import torch
import torch.nn as nn
import reader

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        #conv NN
        self.conv=torch.nn.Sequential(
            #beginning with 6*8*8 input
            torch.nn.Conv2d(6,4,1), #6-channel 8*8 input, 8*8. 4-channel output should be enough. 
            torch.nn.ReLU(),
            #the last layer should generate a 4-channel 8*8 output.
            torch.nn.Conv2d(4,1,4), #4-channel 8*8 input, 4*4 kernel. 4-channel output.
            #dilation seems reasonable. consider adding layer
            torch.nn.ReLU(),
            #the last layer should generate 4-channel 4*4 output
            torch.nn.AvgPool2d(2) #2*2 pooling kernel. just a random guess choice.
            #now should have 4-channel 3*3 output
        )

        #linear layer to generate a single output
        self.lin=torch.nn.Sequential(
            torch.nn.Linear(4,8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,2)
        )
    def forward(self,x):
        '''
        x: 6*8*8
        '''
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.lin(x)
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
        print(ending, prediction)
        loss = loss_fn(prediction,ending)
        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

def train(sets, testing_index):
    '''
    leaving out sets[testing_index] as testing set and train model over everything else
    '''
    model = NeuralNet()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params = model.parameters(),lr = 10)

    for i in range(len(sets)):
        if i == testing_index:
            continue
        print("training on set", i)
        train_set(sets[i],model,loss_fn,optimizer)

    return model, loss_fn, optimizer

def main():
    #white,black,sets = reader.readtxt()
    #model, loss_fn, optimizer = train(sets,0)
    return None
main()


        

