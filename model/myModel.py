import torch
import torchvision.models
import torch.nn as nn
import timm
from torchsummary import summary
import math
import torch.nn.functional as F
from torchsummary import summary
from model.Attention import SE_Block, ECA_Module

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, out_1, out_2, Y):
        # torch.FloatTensor(out_1)
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean(Y * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow
                                      (torch.clamp(self.margin - euclidean_distance.float(), min=0.0), 2))
        return loss_contrastive

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s = 10, m = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

class myresNet(nn.Module):
    def __init__(self, num_classes):
        super(myresNet, self).__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True)
        #self.backbone.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        in_features = self.backbone.fc.in_features
        #out_features = self.backbone.fc.out_features
        self.backbone.fc = nn.Sequential()
        self.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):

        x = self.backbone(x)
        features = x
        #print(features.shape)
        out = self.fc(x)
        return out, features


class myRepVggNet(nn.Module):
    def __init__(self, num_classes):
        super(myRepVggNet, self).__init__()
        self.backbone = timm.create_model("repvgg_b2", pretrained=False)
        self.backbone.load_state_dict(torch.load(r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/preState/repvgg_b2-25b7494e.pth"))

        #self.backbone.stem.conv_kxk.conv = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.backbone.stem.conv_1x1.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)

        self.in_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Sequential()
        self.fc = nn.Linear(self.in_features, num_classes)
        #self.margin = ArcModule(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        x = self.fc(features)

        return x, features

    def get_features_dim(self):
        return self.in_features

class myAttentionRepVggNet(nn.Module):
    def __init__(self, num_classes):
        super(myAttentionRepVggNet, self).__init__()
        self.backbone = timm.create_model("repvgg_b2", pretrained=False)
        self.backbone.load_state_dict(torch.load(r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/preState/repvgg_b2-25b7494e.pth"))

        #self.backbone.stem.conv_kxk.conv = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #self.backbone.stem.conv_1x1.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)

        self.in_features = self.backbone.head.fc.in_features

        self.backbone.head.fc = nn.Sequential()
        self.fc = nn.Linear(self.in_features, num_classes)
        self.AvgPooling = nn.AdaptiveAvgPool2d(1)

        self.se_block = SE_Block(self.in_features)
        self.eca_blcok = ECA_Module(self.in_features)

    def forward(self, x):
        features = self.backbone(x)

        features = self.se_block(features)

        #features = self.AvgPooling(features)
        #features = torch.flatten(features, start_dim=1)

        x = self.fc(features)

        return x, features

    def get_features_dim(self):
        return self.in_features

class myVgg19(nn.Module):
    def __init__(self, num_classes):
        super(myVgg19, self).__init__()
        self.backbone = torchvision.models.vgg19(pretrained=True)
        self.backbone.classifier = nn.Sequential()
        self.features_extracter = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),

        )
        self.fc = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    def forward(self, x):
        x = self.backbone(x)
        features = self.features_extracter(x)
        out = self.fc(features)

        return out, features

class myMobilieNet(nn.Module):
    def __init__(self, num_classes):
        super(myMobilieNet, self).__init__()
        self.backbone = torchvision.models.mobilenet_v3_large(pretrained=True)
        #self.backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Sequential()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes,bias=True)

        )
        # self.classifierForSmall = nn.Sequential(
        #     nn.Linear(in_features=576, out_features=1024, bias=True),
        #     nn.Hardswish(),
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(in_features=1024, out_features=num_classes,bias=True)
        #
        # )
    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)
        #out = self.classifier(x)

        return out, x

class myEfficientNet(nn.Module):
    def __init__(self,num_classess):
        super(myEfficientNet, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b7",pretrained=False)
        self.backbone.load_state_dict(torch.load(r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/preState/tf_efficientnet_b7_ra-6c08e654.pth"))
        self.backbone.classifier = nn.Sequential()
        self.fc = nn.Linear(in_features=2560, out_features=num_classess, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)

        return out, x


if __name__ == '__main__':

    #model = timm.create_model("tf_efficientnet_b7",pretrained=False)
    # for item in timm.list_models():
    #     if "eff" in item:
    #         print(item)
    #model = myresNet(2)#.to(torch.device("cuda"))
    #model = timm.create_model("resnet50", pretrained=True)#.to(torch.device("cuda"))

    #summary(model, (3, 224, 224))
    model = myMobilieNet(2)  #.to(device=torch.device("cuda"))
    #model.load_state_dict(torch.load(r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/MobileNetV3LargeOut/1/0.949999988079071_epoch59normal_acc0.9399999976158142ossc_acc0.9599999785423279.pth"))
    #model = timm.create_model("repvgg_b2")
    #model.head.fc = nn.Sequential()
    #print(rep_model)
    print("******************************")
    #print(model(torch.FloatTensor(1, 3, 224, 224)).shape)
    #print(model.classifier)   #summary(rep_model, (3, 224, 224))
    #print(*list(model.backbone.children())[0])
    features_extracter = nn.Sequential(*(list(model.backbone.children())[:-2]))
    features_extracter = nn.Sequential(*list(features_extracter.children()))
    f = list(*nn.Sequential(*(list(model.backbone.children())[:-2])))
    print(f[0])
    #print(*list(model.children())[:-1])

    #summary(model, (3, 224, 224))
    torch.cuda.empty_cache()