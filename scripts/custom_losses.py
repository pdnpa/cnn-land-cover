## Custom losses

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl

class FocalLoss_2(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets 
    https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b'''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class FocalLoss(nn.Module):
    '''https://github.com/amirhosseinh77/UNet-AerialSegmentation/blob/main/losses.py'''
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class mIoULoss(nn.Module):
    '''https://github.com/amirhosseinh77/UNet-AerlambdaialSegmentation/blob/main/losses.py'''
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n,h,w = tensor.size()
        one_hot = torch.zeros(n,self.classes,h,w).to(tensor.device).scatter_(1,tensor.view(n,1,h,w),1)
        return one_hot

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs,dim=1)
        
        # Numerator Product
        target_oneHot = self.to_one_hot(target)
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)

        #Denominator 
        union= inputs + target_oneHot - (inputs*target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)

        loss = inter/union

        ## Return average loss over classes and batch
        return 1-loss.mean()

class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me) 

# class LossCurveCallback(pl.Callback):
#         '''PyTorch Lightning metric callback.
#     I think it is very clunky to make a callback class just to plot loss curve
#     but I guess that's what you get if you want lightning to tbe minimalist
#     https://forums.pytorchlightning.ai/t/how-to-access-the-logged-results-such-as-losses/155
#     '''

#     def __init__(self, lnp, figsize=[15, 10]):
#         super().__init__()
#         self.lnp = lnp
#         self.figsize = figsize

#     def on_init_end(self, trainer):
#         self.a_trn_loss = np.ones(trainer.max_epochs) * np.inf
#         self.a_val_loss = np.ones(trainer.max_epochs) * np.inf

#     def on_validation_epoch_end(self, trainer, pl_module):
#         self.lnp.lnp('EPOCH' +' '+ str(trainer.current_epoch) +' '+
#               'sanity_checking' +' '+ str(trainer.sanity_checking) +' '+
#               str(trainer.callback_metrics))
#         if not trainer.sanity_checking: # WARN: sanity_check is turned on by default
#             self.a_trn_loss[trainer.current_epoch] = trainer.callback_metrics['trn_loss']
#             self.a_val_loss[trainer.current_epoch] = trainer.callback_metrics['val_loss']

#     def on_train_end(self,trainer,pl_module):
#         '''TODO: hp_metric won't update at on_fit_end
#         If I use on_fit_end and:
#         If I use self.log(), I will get MisconfigurationException
#         If I use add_scalar(), It won't update, it will stay at -1
#         '''
#         '''Note: on_fit_end was destroyed at on_fit_end, but on_train_end
#         pl_module.log('minval', self.a_val_loss.min())
#         pl_module.log('hp_metric', self.a_val_loss.min())
#         pl_module.log('aminval', self.a_val_loss.argmin())
#         MisconfigurationException: on_fit_end function doesn't support logging using `self.log()`
#         '''
#         trainer.logger[0].experiment.add_scalar('aminval', self.a_val_loss.argmin())
#         trainer.logger[0].experiment.add_scalar('minval', self.a_val_loss.min())
#         trainer.logger[1].experiment.summary['aminval'] = self.a_val_loss.argmin()
#         trainer.logger[1].experiment.summary['minval'] = self.a_val_loss.min()
#         # trainer.logger.experiment.add_scalar('hp_metric', self.a_val_loss.min())
#         self.lnp.lnp('aminval ' + str(self.a_val_loss.argmin()))
#         self.lnp.lnp('minval ' + str(self.a_val_loss.min()))
#         # self.lnp.lnp('hp_metric ' + str(self.a_val_loss.min()))
#         ''' hp_metric
#         This the metric that Tensorboard will use to compare between runs to pick the best hyperparameters.
#         Ideally, it is a single scalar number per run.
#         '''

#         f,a = plt.subplots(figsize=self.figsize)
#         a.set_title('Loss curve')
#         a.plot(self.a_trn_loss, label='trn_loss')
#         a.plot(self.a_val_loss, label='val_loss')
#         # TODO: twinx and plot LR. Maybe we can see drop when the LR drop
#         # TODO: twinx and plot LR. Maybe plot gradient l2 norm sum
#         a.set_xlabel('Epoch')
#         a.set_ylabel('Loss')
#         a.vlines(x=self.a_val_loss.argmin(), ymin=self.a_val_loss.min(), ymax=self.a_val_loss[:trainer.current_epoch].max(),
#                  label='lowest validation = '+str(self.a_val_loss.min())+' at '+str(self.a_val_loss.argmin()))
#         a.legend()
#         trainer.logger[0].experiment.add_figure('loss_curve', f)
#         trainer.logger[1].experiment.log({'loss_curve': f})
        
#         f,a = plt.subplots(figsize=self.figsize)
#         a.set_title('Loss curve')
#         a.semilogy(self.a_trn_loss, label='trn_loss')
#         a.semilogy(self.a_val_loss, label='val_loss')
#         a.set_xlabel('Epoch')
#         a.set_ylabel('Loss')
#         a.vlines(x=self.a_val_loss.argmin(), ymin=self.a_val_loss.min(), ymax=self.a_val_loss[:trainer.current_epoch].max(),
#                  label='lowest validation = '+str(self.a_val_loss.min())+' at '+str(self.a_val_loss.argmin()))
#         a.legend()
#         trainer.logger[0].experiment.add_figure('loss_curve_log', f)
#         trainer.logger[1].experiment.log({'loss_curve_log': f})