from impfun import *

class convNet_2(nn.Module):
    def __init__(self):
        super(convNet_2, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, 6)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(57*8, 128)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 1)
        )
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class convNet_2_1fc(nn.Module):
    def __init__(self):
        super(convNet_2_1fc, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, 6)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(57*8, 1)
        )
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
class convNet_2_act_1fc(nn.Module):
    def __init__(self):
        super(convNet_2_act_1fc, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.AdaPool = nn.AdaptiveMaxPool1d(100)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, 6),
            nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(57*8,1)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
   


class convNet_2_act(nn.Module):
    def __init__(self):
        super(convNet_2_act, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.AdaPool = nn.AdaptiveMaxPool1d(100)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 8, 6),
            nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(57*8, 128)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features





class convNet_3(nn.Module):
    def __init__(self):
        super(convNet_3, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 6)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 6)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(57*8, 128)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_3_act(nn.Module):
    def __init__(self):
        super(convNet_3_act, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 6),
             nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 6),
             nn.RReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(57*8, 128)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_4(nn.Module):
    def __init__(self):
        super(convNet_4, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 6)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 6)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 10, 6)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(57*10, 128)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_4_act(nn.Module):
    def __init__(self):
        super(convNet_4_act, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 6),
            nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 6),
            nn.RReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 10, 6),
            nn.RReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(57*10, 128)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_5(nn.Module):
    def __init__(self):
        super(convNet_5, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 6)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 6)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 10, 6)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(10, 12, 6)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(57*12, 128)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.layer5(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class convNet_5_act(nn.Module):
    def __init__(self):
        super(convNet_5_act, self).__init__()
        self.spp = SpatialPyramidPooling('max')
        self.pool =  nn.MaxPool1d(2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 4, 9),
            nn.RReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 6, 6),
            nn.RReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 8, 6),
            nn.RReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(8, 10, 6),
            nn.RReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(10, 12, 6),
            nn.RReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(57*12, 128)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.layer5(x)
        x = self.spp(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.layer6(x)
        x = self.layer7(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
model_list = []
model_class = [convNet_2(), convNet_2_act(), convNet_3(), convNet_3_act(), convNet_4(), convNet_4_act(), 
               convNet_5(), convNet_5_act(),convNet_2_1fc,convNet_2_1fc]
for model in model_class:
    model_list.append(model.to(device))