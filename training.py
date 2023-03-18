import torch
import torch.nn as nn
import random
import reader
import dataloader
import torch
from torch.utils.data import Dataset, DataLoader

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        #conv NNdataloaders
        self.conv=torch.nn.Sequential(
            #beginning with 6*8*8 input
            torch.nn.Conv2d(6,18,2), #6-channel 8*8 input--2*2 kernel-->7*7 output. using 4-channel. 
            torch.nn.ReLU(),
            torch.nn.Conv2d(18,1,4), #4-channel 7*7 input-->4*4 kernel->4*4 output. 1-channel output.
            #dilation seems reasonable. consider adding layer
            torch.nn.ReLU(),
            #the last layer should generate 4-channel 4*4 output
            torch.nn.AvgPool2d(2) #2*2 pooling kernel. just a random guess choice.
            
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

def train_set(dataloader,model,loss_fn,optimizer):
    '''
    trains model over a given set with specified loss_fn and optimizer.
    '''
    for board,ending in dataloader:
        #print(board,ending)
        prediction = model(board)
        prediction = torch.tensor([float(prediction[0][0]),float(prediction[0][1])])
        ending_tensor = torch.tensor([float(ending[0]),float(ending[1])])
        #print(prediction)
        loss = loss_fn(prediction,ending_tensor)
        optimizer.zero_grad()
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

def train_loop(dataloaders,epochs=10):
    model=NeuralNet()
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=1)
    print("training...")
    for epoch in range(epochs):
        test_set_number = random.randint(0,9)
        print(" training loop:",epoch)
        print(" testing set is: set",test_set_number)
        train_sets(dataloaders,test_set_number,model,loss_fn,optimizer)
        print(" training finished, now testing...")
        test_set(dataloaders[test_set_number],test_set_number,model,loss_fn)
    return model, loss_fn, optimizer

def train_sets(dataloaders,test_set_number,model,loss_fn,optimizer):
    for i in range(len(dataloaders)):
        if i == test_set_number:
            print("  bypass testing set, index =", i)
            continue
        print("  training on set", i)
        train_set(dataloaders[i],model,loss_fn,optimizer)

def test_set(test_loader,test_set_number,model,loss_fn):
    print("  testing on testing set, index =", test_set_number)
    test_loss = 0
    for board,ending in test_loader:
        with torch.no_grad():
            prediction = model(board)
            prediction = torch.tensor([float(prediction[0][0]),float(prediction[0][1])])
            ending_tensor = torch.tensor([float(ending[0]),float(ending[1])])
            loss=loss_fn(prediction,ending_tensor)
            test_loss+=loss
    test_loss = test_loss / len(test_loader)
    print("  average test loss on set:",test_loss)
    
def init(sets):
    datasets = []
    dataloaders = []
    for i in range(10):
        datasets.append(dataloader.BoardDataset(sets[i][0],sets[i][1]))
        dataloaders.append(torch.utils.data.DataLoader(datasets[i], batch_size=1, shuffle=True))
    return datasets,dataloaders 
    

def main():
    #white,black,sets = reader.readtxt()
    #model, loss_fn, optimizer = train(sets,0)
    return None
main()


        

        

