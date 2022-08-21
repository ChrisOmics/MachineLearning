
##christopher Robles

batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

torch.manual_seed(0)


## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_data = datasets.MNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
#get data with labels 0 or 1
subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().view(-1)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))


subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().view(-1)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))


# Same as linear regression! 
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class HingeLoss(nn.Module):

    def __init__(self, size_average=True):
        super(HingeLoss,self).__init__()
        self.size_average = size_average

    def forward(self,output,y):
        loss=1-y*output
        loss[loss<0]=0 # set negative losses to 0 (max function)

        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]
        return loss
    
def run_model(num_epochs, learning_rate, loss_function, momentum):
    print('Model: {}, Learning Rate: {}, Momentum: {}.'.format(loss_function, learning_rate, momentum))
    
    input_dim = 28*28
    output_dim = 1   
    model = LogisticRegressionModel(input_dim, output_dim)
    if (loss_function == 'logistic'):
        criterion = nn.SoftMarginLoss()
    else:
        criterion = HingeLoss()    
    #run with and without momentem (between .9 and 0.99)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=momentum)
    # Training the Model
    # Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.
#    num_epochs=10
    epoch_iter = 0
    correct=0
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        total_loss = []
        correct=0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            #Convert labels from 0,1 to -1,1
            labels = Variable(2*(labels.float()-0.5))

            ## TODO 
            optimizer.zero_grad()
            outputs = model(images)
            # Calculate Loss
            loss = criterion(outputs.squeeze(1), labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            
            total_loss.append(torch.Tensor.item(loss))
            e_loss=sum(total_loss)/(len(total_loss))
        print('Epoch: {}. Loss: {}. '.format(epoch_iter, e_loss))

        epoch_iter+=1
        #Print your accuracy results every epoch
    total=0
    correct=0
    for images, labels in test_loader:
        # Load images to a Torch Variable
        images = images.view(-1, 28*28)

        # Forward pass only to get logits/output
        outputs = model(images)
        if loss_function == 'logistic':
            sig = nn.Sigmoid() # apply sigmoid to linear function
            predicted = torch.round(sig(outputs.data)) 
        else:
            predicted = torch.sign(outputs.data) 
            predicted[predicted < 0] = 0
        # Total number of labels
        total += images.shape[0]
        # Total correct predictions
        correct += (predicted.view(-1).long() == labels).sum().float()
    accuracy = 100 * correct.float() / total

    print('Accuracy: {}'.format(accuracy.float()))
    # Test the Model

#run_model(num_epochs, learning_rate, loss_function, momentum):
run_model(10, 0.001, "logistic", 0)
run_model(10, 0.001, "logistic", 0.9)
run_model(10, 0.001, "SVM", 0)
run_model(10, 0.001, "SVM", 0.9)


##this is adjusting the learning rate
#run_model(num_epochs, learning_rate, loss_function, momentum):
run_model(10, 0.1, "logistic", 0)
run_model(10, 0.1, "logistic", 0.9)
run_model(10, 0.1, "SVM", 0)
run_model(10, 0.1, "SVM", 0.9)
