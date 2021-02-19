## import package
#from function import *
import sys
import os
from sklearn import preprocessing
from early_stop import *
from function import *

## load the realating csv file
# get command line argument length.
#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/cv/sequenceID/testFolds/6"#sys.argv[1]
dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic/cv/sequenceID/testFolds/6"
## load the realating csv file
dir_path_split = dir_path.split("cv")
fold_path_split = dir_path.split("/testFolds/")
inputs_path = dir_path_split[0] + "inputs.csv.xz"
labels_path = dir_path_split[0] + "outputs.csv.xz"
folds_path = fold_path_split[0] + "/folds.csv"
fold_num = int(fold_path_split[1])
outputs_path = dir_path + "/randomTrainOrderings/1/models/"

inputs = pd.read_csv(inputs_path)
inputs = inputs.replace([np.inf, -np.inf], np.nan)
inputs = inputs.dropna(axis=1, how='any')
labels = pd.read_csv(labels_path)
folds = pd.read_csv(folds_path)

## procssing data
labels = labels.values
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
Scale_inputs = preprocessing.scale(inputs.iloc[:, 1:])
Scale_inputs = pd.concat([seq_id, pd.DataFrame(Scale_inputs)], axis=1)
Scale_inputs = np.array(Scale_inputs)

folds = np.array(folds)
_, cor_index = np.where(Scale_inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

## define the linear network
class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(num_feature, 1)

    def forward(self, x):
        x =  self.fc1(x)
        return x

## define the loss funciton and init the model
inputs = Scale_inputs
criterion = SquareHingeLoss()
step = 1e-3
epoch = 3000
model_list = []
optimizer_list = []
model = LinearNN().to(device)
optimizer = optim.Adam(model.parameters(),  lr= step)

#save the init model
if not os.path.exists("model_path/" + dir_path):
    os.makedirs("model_path/" + dir_path) 

PATH = "model_path/" + dir_path + 'cifar_net.pth'
torch.save(model.state_dict(), PATH)

# transfer data type
inputs, labels = Typetransfer_2D(inputs, labels)

# get best epoch
train_data, test_data, train_label, test_label = SplitFolder(inputs, labels, 
                                                    folds_sorted[:, 1], fold_num)

best_epoch = earlyStop(model, optimizer, criterion, train_data, train_label, epoch).__call__()
num_test = test_data.shape[0]

# init variables
model = LinearNN().to(device)
model.load_state_dict(torch.load(PATH))
optimizer = optim.Adam(model.parameters(),  lr= step)
    
_, test_outputs = Full(model, optimizer, criterion, 
                           train_data, train_label, test_data, 
                           test_label, best_epoch).__call__()

# test data
with torch.no_grad():
    accuracy = 0
    for index in range(num_test):
        accuracy = accuracy + Accuracy(test_outputs[index], test_label[index].cpu().data.numpy())
    print(accuracy/num_test * 100)

# this fucntion output the csv file
cnn_output = pd.DataFrame(test_outputs.cpu().data.numpy())
if not os.path.exists(outputs_path + "Cnn_early_pytorch"):
    os.mkdir(outputs_path + "Cnn_early_pytorch") 
cnn_output.to_csv(outputs_path + 'Cnn_early_pytorch/predictions.csv')  



        
         

