from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=0.5, gamma=2, use_alpha=True, size_average=True, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha).float()
        elif isinstance(alpha, (int, float)):
            assert alpha < 1
            self.alpha = torch.zeros(class_num)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 
            assert len(self.alpha)==class_num  # alpha=[1,1,1,...] <--> alpha=0.5 <--> no weight
        self.softmax = nn.Softmax(dim=1)
        self.size_average = size_average
        
        self.alpha = self.alpha / self.alpha.sum() * class_num # normalization
        
    def forward(self, pred, target):
        if isinstance(self.ignore_index, int) and self.ignore_index<self.class_num:
            mask = (target != self.ignore_index)
            target = target.masked_select(mask)
            pred = pred.masked_select(mask.unsqueeze(1)).view(-1, self.class_num)
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=1e-9,max=1.0)
        
        target_ = torch.zeros(target.size(0),self.class_num)
        target_.scatter_(1, target.view(-1, 1), 1.)
        
        prob_log = prob.log() #+ 1e-10
        batch_loss = - self.alpha * torch.pow(1-prob,self.gamma) * prob_log * target_
        #batch_loss = - torch.pow(1-prob,self.gamma) * prob_log * target_
        batch_loss = batch_loss.sum(dim=1)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
    
    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        elif self.reduction == 'mean':
            return torch.mean(loss)
        
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

 
if __name__ == '__main__':
    label = torch.tensor([0])
    pred = torch.tensor([[ 0.1094,  0.0912,  0.1766]])
    pred2 = torch.tensor([[1,0,0]]).float()

    loss_fn = FocalLoss(alpha=[1, 1, 1], gamma=0, class_num=3)
    print("alpha:", loss_fn.alpha)
    loss_1 = loss_fn(pred, label)
    print(loss_1)
    loss_2 = loss_fn(pred2, label)
    print(loss_2)

    
    
    
    
    
    
    
    