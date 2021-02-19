
## import package
from impfun import *
import sys
from sklearn import preprocessing

## load the realating csv file
# get command line argument length.
#argv = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/
#argv = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic/cv/sequenceID/"

## load the realating csv file
#dir_path = argv #+ '/Inputs/'
#inputs_file = 'inputs.csv.xz'
#outputs_file = 'outputs.csv.xz'
#folds_file = 'folds.csv'


#dir_path = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic"

## load the realating csv file
# get command line argument length.
argv = "/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/systematic/"#sys.argv[1]

## load the realating csv file
dir_path = argv #+ '/Inputs/'
inputs_file = 'inputs.csv.xz'
outputs_file = 'outputs.csv.xz'
folds_file = 'folds.csv'

inputs = pd.read_csv(dir_path + inputs_file)
outputs = pd.read_csv(dir_path + outputs_file)
folds = pd.read_csv(dir_path + "cv/sequenceID/" + folds_file)

## procssing data
labels = outputs.values
num_id = labels.shape[0]
num_feature = inputs.shape[1] - 1
seq_id = inputs.iloc[:, 0].to_frame()
min_max_scaler = preprocessing.MinMaxScaler()
Scale_inputs = preprocessing.scale(inputs.iloc[:, 1:])
Scale_inputs = pd.concat([seq_id, pd.DataFrame(Scale_inputs)], axis=1)
Scale_inputs = np.array(Scale_inputs)

folds = np.array(folds)
_, cor_index = np.where(Scale_inputs[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

# build the net work
class convNet(nn.Module):
    def __init__(self):
        super(convNet, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 5, 9), #1 1 117 -> 1 5 109
        )
        self.layer2 = nn.Sequential(
            nn.Linear(5*21, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = SquareHingeLoss()
inputs = Scale_inputs

model_list = []
optimizer_list = []
for num_model in range(6):
    model = convNet().to(device)
    optimizer = optim.Adam(model.parameters(),  lr= 1e-5)
    model_list.append(model)
    optimizer_list.append(optimizer)

# split train test data, using Kfold
cnn_test_acc = []
best_output_list = []
for fold_num in range(1, 7):
    train_data, test_data, train_label, test_label = SplitFolder(inputs, labels, 
                                                    folds_sorted[:, 1], fold_num)

    # split train vlidation data
    num_sed_fold = train_data.shape[0]
    sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
    left = np.arange(num_sed_fold % 5) + 1
    sed_fold = np.concatenate((sed_fold, left), axis=0)
    np.random.shuffle(sed_fold)

    subtrain_data, valid_data, subtrain_label, valid_label = SplitFolder(train_data, train_label,             
                                                                         sed_fold, 2)
    # set up model
    num_model = fold_num - 1
    model = model_list[num_model]
    optimizer = optimizer_list[num_model]

    # transfer data
    num_train = subtrain_data.shape[0]
    num_valid = valid_data.shape[0]
    num_test = test_data.shape[0]
    channel = 1
    subtrain_data, subtrain_label = Typetransfer_3D(subtrain_data, subtrain_label, channel)
    valid_data, valid_label = Typetransfer_3D(valid_data, valid_label, channel)
    test_data, test_label = Typetransfer_3D(test_data, test_label, channel)

    # init variables
    step = 0
    train_losses, valid_losses, valid_accuracy= [], [], []
    parameters = []
    test_outputs = []
    cnn_test_accuracy = []
    num_epoch = 7
    mini_batches = 5 

    ## train the network
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        for index in range(num_train):
            model.train()
        
            # init variable
            train_loss = 0
            valid_loss = 0     
            accuracy = 0

            # step + 1
            step += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # do ADM
            outputs = model(subtrain_data[index].unsqueeze(0))
        
            loss = criterion(outputs, subtrain_label[index])
            loss.backward()
        
            optimizer.step()
        
            if step % mini_batches == 0:
                with torch.no_grad():
                    model.eval()
        
                    # calculate the loss of train and valid
                    train_outputs = model(subtrain_data)
                    train_loss = criterion(train_outputs, subtrain_label)
                    train_losses.append(train_loss.cpu().data.numpy())
        
                    valid_outputs = model(valid_data)
                    valid_loss = criterion(valid_outputs, valid_label)
                    valid_losses.append(valid_loss.cpu().data.numpy())
        
                test_output = model(test_data)
                test_outputs.append(test_output.cpu().data.numpy())

    # choose the min value from valid list
    min_loss_train = min(train_losses)
    min_train_index = train_losses.index(min(train_losses))
    min_loss_valid = min(valid_losses)
    best_parameter_value = valid_losses.index(min(valid_losses))
    best_output = test_outputs[best_parameter_value]
    best_output_list.append(best_output)

    # plot
    plt.plot(train_losses, label = 'Training loss')
    plt.plot(valid_losses, label = 'Validation loss')
    plt.scatter(min_train_index, min_loss_train, label = 'min train value', color='green')
    plt.scatter(best_parameter_value, min_loss_valid, label = 'min valid value', color='black')
    plt.legend(frameon=False)
    plt.xlabel("step of every 10 min-bath")
    plt.ylabel("loss")
    plt.show()
    
    # test data
    with torch.no_grad():
        accuracy = 0
        for index in range(num_test):
            accuracy = accuracy + Accuracy(best_output[index], test_label[index].cpu().data.numpy())
        cnn_test_accuracy.append(accuracy/num_valid * 100)

cnn_output = pd.DataFrame(best_output_list[0])
cnn_output = OutputFile(cnn_output, best_output_list)
cnn_output.to_csv(argv + '/Outputs/cnnModel.csv', index = None, header = False) 