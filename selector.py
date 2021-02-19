from function import *

class Selector:
    def __init__(self, model, PATH, criterion, train_data, train_label, epoch, rate_list):
        self.model = model
        self.PATH = PATH
        self.criterion = criterion
        self.train_data = train_data
        self.train_label = train_label
        self.epoch = epoch
        self.rate_list = rate_list
    
    def split_data(train_data, train_label):
        num_sed_fold = train_data.shape[0]
        sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
        left = np.arange(num_sed_fold % 5) + 1
        sed_fold = np.concatenate((sed_fold, left), axis=0)
        np.random.shuffle(sed_fold)

        subtrain_data, valid_data, subtrain_label, valid_label = SplitFolder(train_data, 
                                                                             train_label, sed_fold, 1)
        
        return subtrain_data, valid_data, subtrain_label, valid_label 

    def __call__(self):
        # split to subtrain, valid
        subtrain_data, valid_data, subtrain_label, valid_label = Selector.split_data(self.train_data, 
                                                                                      self.train_label)
        
        # transfer data type
        num_train = subtrain_data.shape[0]
        num_valid = valid_data.shape[0]
        subtrain_data, subtrain_label = Typetransfer_2D(subtrain_data, subtrain_label)
        valid_data, valid_label = Typetransfer_2D(valid_data, valid_label)
        
        # init variable
        train_loss_list = []
        valid_loss_list = []

        for rate in self.rate_list:
            # trian the model
            # init model
            self.model.load_state_dict(torch.load(self.PATH))

            valid_loss = Selector.__train__(subtrain_data, subtrain_label, 
                                            valid_data, valid_label, 
                                            self.model, self.criterion, 
                                            rate, self.epoch)
            
            valid_loss_list.append(valid_loss)
        
        min_loss_valid = min(valid_loss_list)
        best_index = valid_loss_list.index(min_loss_valid)
        best_rate = self.rate_list[best_index]
        
        return best_rate

    def __train__(subtrain_data, subtrain_label, valid_data, 
                  valid_label, model, criterion, rate, epoch):       
        # init variables
        step = 0
        valid_losses = []
        optimizer = optim.Adam(model.parameters(),  lr= 1e-4)
        
        ## train the network
        for epoch in range(epoch):  # loop over the dataset multiple times
            for index, (data, label) in enumerate(zip(subtrain_data, subtrain_label)):
                model.train()   
        
                # step + 1
                step += 1
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # do SGD
                outputs = model(data)
                loss = criterion(outputs, label)
                loss = L1Regularizer(model, loss, rate).regularized_param()
                
                loss.backward()
                optimizer.step()
            
        for index, (data, label) in enumerate(zip(valid_data, valid_label)):
            with torch.no_grad():
                model.eval()
                valid_outputs = model(data)                    
                valid_loss = criterion(valid_outputs, label)
                valid_loss = L1Regularizer(model, valid_loss, rate).regularized_param()
                valid_losses.append(valid_loss.cpu().data.numpy())
            
        valid_loss = np.average(valid_losses)
        
        return valid_loss
    
        
    
    
                  
