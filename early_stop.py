from full import*

class earlyStop:
    def __init__(self, model, optimizer, criterion, train_data, train_label, epoch):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data
        self.train_label = train_label
        self.epoch = epoch
    
    def split_data(train_data, train_label):
        num_sed_fold = train_data.shape[0]
        sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
        left = np.arange(num_sed_fold % 5) + 1
        sed_fold = np.concatenate((sed_fold, left), axis=0)
        np.random.shuffle(sed_fold)

        subtrain_data, valid_data, subtrain_label, valid_label = SplitFolder(train_data, train_label, sed_fold, 1)
        
        return subtrain_data, valid_data, subtrain_label, valid_label 
    
    def paramter_select(avg_valid_losses):
        min_loss_valid = min(avg_valid_losses)
        best_parameter = avg_valid_losses.index(min_loss_valid)
        
        return best_parameter

    def __call__(self):
        # split to subtrain, valid
        subtrain_data, valid_data, subtrain_label, valid_label = earlyStop.split_data(self.train_data, self.train_label)
        
       
        # train data with full gradient descent
        avg_valid_losses ,_ = Full(self.model, self.optimizer, self.criterion, 
                                             subtrain_data, subtrain_label, valid_data, 
                                             valid_label, self.epoch).__call__()

        #get best epoch and return 
        best_epoch = earlyStop.paramter_select(avg_valid_losses)
    
        return best_epoch
               