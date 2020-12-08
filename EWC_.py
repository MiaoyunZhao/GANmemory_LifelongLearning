import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
from torch import autograd
import numpy as np
import torchvision.models as models

# res18 = models.resnet18(pretrained=True)


# class Net(nn.Module):
#     def __init__(self, nlabels=1000, device=None):
#         super(Net, self).__init__()
#         self.device = device
#
#         #### MODEL PARAMS ####
#         self.feat = models.resnet18(pretrained=True)
#         self.feat.fc = nn.Linear(512, nlabels)
#         # self.clasify = nn.Sequential(
#         #     nn.Linear(25088, 4096),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p=0.5, inplace=False),
#         #     nn.Linear(4096, 1000),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p=0.5, inplace=False),
#         #     nn.Linear(1000, nlabels),
#         # )
#         self.Fisher = {}
#         # Self Params
#         self.params = [param for param in self.feat.parameters()]
#
#         # -EWC:
#         self.ewc_lambda = 0  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
#         self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
#         self.online = True  # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
#         self.fisher_n = None  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
#         self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
#         self.EWC_task_count = 0  # -> keeps track of number of quadratic loss terms (for "offline EWC")
#
#     def forward(self, x):
#         h = self.feat(x)
#         return h
#
#     def estimate_fisher(self, data_loader, allowed_classes=None, collate_fn=None):
#         '''After completing training on a task, estimate diagonal of Fisher Information matrix.
#
#         [dataset]:          <DataSet> to be used to estimate FI-matrix
#         [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''
#
#         # Prepare <dict> to store estimated Fisher Information matrix
#         est_fisher_info = {}
#         for n, p in self.named_parameters():
#             if p.requires_grad:
#                 n = n.replace('.', '__')
#                 est_fisher_info[n] = p.detach().clone().zero_()
#
#         # Set model to evaluation mode
#         mode = self.training
#         self.eval()
#
#         # # Create data-loader to give batches of size 1
#         # data_loader = utils.get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)
#
#         # Estimate the FI-matrix for [self.fisher_n] batches of size 1
#         for index,(x,y) in enumerate(data_loader):
#             # break from for-loop if max number of samples has been reached
#             if self.fisher_n is not None:
#                 if index >= self.fisher_n:
#                     break
#             # run forward pass of model
#             x = x.to(self._device())
#             output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
#             if self.emp_FI:
#                 # -use provided label to calculate loglikelihood --> "empirical Fisher":
#                 label = torch.LongTensor([y]) if type(y)==int else y
#                 if allowed_classes is not None:
#                     label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
#                     label = torch.LongTensor(label)
#                 label = label.to(self._device())
#             else:
#                 # -use predicted label to calculate loglikelihood:
#                 label = output.max(1)[1]
#             # calculate negative log-likelihood
#             negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
#
#             # Calculate gradient of negative loglikelihood
#             self.zero_grad()
#             negloglikelihood.backward()
#
#             # Square gradients and keep running sum
#             for n, p in self.named_parameters():
#                 if p.requires_grad:
#                     n = n.replace('.', '__')
#                     if p.grad is not None:
#                         est_fisher_info[n] += p.grad.detach() ** 2
#
#         # Normalize by sample size used for estimation
#         est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}
#
#         # Store new values in the network
#         for n, p in self.named_parameters():
#             if p.requires_grad:
#                 n = n.replace('.', '__')
#                 # -mode (=MAP parameter estimate)
#                 self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count+1),
#                                      p.detach().clone())
#                 # -precision (approximated by diagonal Fisher Information matrix)
#                 if self.online and self.EWC_task_count==1:
#                     existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
#                     est_fisher_info[n] += self.gamma * existing_values
#                 self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count+1),
#                                      est_fisher_info[n])
#
#         # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
#         self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1
#
#         # Set model back to its initial mode
#         self.train(mode=mode)
#
#     def ewc_loss(self):
#         '''Calculate EWC-loss.'''
#         if self.EWC_task_count>0:
#             losses = []
#             # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
#             for task in range(1, self.EWC_task_count+1):
#                 for n, p in self.named_parameters():
#                     if p.requires_grad:
#                         # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
#                         n = n.replace('.', '__')
#                         mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
#                         fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
#                         # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
#                         fisher = self.gamma*fisher if self.online else fisher
#                         # Calculate EWC-loss
#                         losses.append((fisher * (p-mean)**2).sum())
#             # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
#             return (1./2)*sum(losses)
#         else:
#             # EWC-loss is 0 if there are no stored mode and precision yet
#             return torch.tensor(0., device=self._device())
#
#     # def estimate_fisher(self, dataset, sample_size, batch_size=32):
#     #     # Get loglikelihoods from data
#     #     self.F_accum = []
#     #     for v, _ in enumerate(self.params):
#     #         self.F_accum.append(np.zeros(list(self.params[v].size())))
#     #     data_loader = dataset
#     #     loglikelihoods = []
#     #
#     #     for x, y in data_loader:
#     #         #print(x.size(), y.size())
#     #         # x = x.view(batch_size, -1)
#     #         x = x.to(self.device)
#     #         y = y.to(self.device)
#     #
#     #         loglikelihoods.append(F.log_softmax(self(x), dim=1)[range(batch_size), y.data])
#     #
#     #         if len(loglikelihoods) >= sample_size // batch_size:
#     #             break
#     #
#     #         #loglikelihood = torch.cat(loglikelihoods).mean(0)
#     #         loglikelihood = torch.cat(loglikelihoods).mean(0)
#     #         loglikelihood_grads = autograd.grad(loglikelihood, self.params,retain_graph=True)
#     #         #print("FINISHED GRADING", len(loglikelihood_grads))
#     #         for v in range(len(self.F_accum)):
#     #             #print(len(self.F_accum))
#     #             torch.add(torch.Tensor((self.F_accum[v])).to(self.device), torch.pow(loglikelihood_grads[v], 2).data)
#     #
#     #     for v in range(len(self.F_accum)):
#     #         self.F_accum[v] /= sample_size
#     #
#     #     parameter_names = [
#     #         n.replace('.', '__') for n, p in self.named_parameters()
#     #     ]
#     #     #print("RETURNING", len(parameter_names))
#     #
#     #     return {n: g for n, g in zip(parameter_names, self.F_accum)}
#
#     # def consolidate(self, fisher):
#     #     for n, p in self.named_parameters():
#     #         n = n.replace('.', '__')
#     #         self.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
#     #         #print(dir(fisher[n].data))
#     #         # self.register_buffer('{}_estimated_fisher'
#     #         #                      .format(n), fisher[n].data)
#     #         self.register_buffer('{}_estimated_fisher'.format(n), torch.tensor(fisher[n]).detach().clone())
#     #
#     # def ewc_loss(self, lamda):
#     #     try:
#     #         losses = []
#     #         for n, p in self.clasify.named_parameters():
#     #             # retrieve the consolidated mean and fisher information.
#     #             n = n.replace('.', '__')
#     #             mean = getattr(self, '{}_estimated_mean'.format(n))
#     #             fisher = getattr(self, '{}_estimated_fisher'.format(n))
#     #             # wrap mean and fisher in Vs.
#     #             # mean = V(mean)
#     #             fisher = fisher.data
#     #             # calculate a ewc loss. (assumes the parameter's prior as
#     #             # gaussian distribution with the estimated mean and the
#     #             # estimated cramer-rao lower bound variance, which is
#     #             # equivalent to the inverse of fisher information)
#     #             losses.append((fisher * (p-mean)**2).sum())
#     #         return (lamda/2)*sum(losses)
#     #     except AttributeError:
#     #         # ewc loss is 0 if there's no consolidated parameters.
#     #         return torch.zeros(1).to(self.device)


# vgg16 = models.vgg16(pretrained=True)
# vgg16 = models.vgg16_bn(pretrained=True)
class Net(nn.Module):
    def __init__(self, nlabels=1000, device=None):
        super(Net, self).__init__()
        self.device = device

        #### MODEL PARAMS ####
        # self.feat = vgg16.features
        # self.clasify = nn.Sequential(
        #     nn.Linear(25088, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(4096, 1000),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(1000, nlabels),
        # )
        # self.params = [param for param in self.clasify.parameters()]

        self.feat = models.resnet18(pretrained=True)
        # self.feat = models.resnet18(pretrained=False)
        self.feat.fc = nn.Linear(512, nlabels)
        # self.params = [param for n, param in self.feat.named_parameters() if (n.find('fc')>=0 or n.find('layer4.1')>=0)]
        # self.params = [param for n, param in self.feat.named_parameters() if
        #                (n.find('fc') >= 0 or n.find('layer4.') >= 0)]
        self.params = [param for n, param in self.feat.named_parameters() if 1]

        self.online=False
        self.gamma = 1.

    def forward(self, x):
        if x.shape[3]>224:
            h = F.interpolate(x, 224, mode='bilinear')
        else:
            h= x
        # h = F.interpolate(x, 224, mode='bilinear')
        h = self.feat(h)
        # h = F.relu(self.EWC_fc1(h))
        # h = self.fc1_drop(h)
        # h = F.relu(self.EWC_fc2(h))
        # h = self.fc2_drop(h)
        # h = self.EWC_fc3(h)
        # h = self.clasify(h.view(x.shape[0],-1))
        return h

    def estimate_fisher(self, dataset, sample_size, batch_size=32, task_id=0):
        # Get loglikelihoods from data
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        mode = self.training
        self.eval()
        data_loader = dataset
        for index,(x,y) in enumerate(data_loader):
            #print(x.size(), y.size())
            # x = x.view(batch_size, -1)
            if index >= sample_size//batch_size:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            loglikelihoods = F.log_softmax(self(x), dim=1)[range(batch_size), y.data].mean()
            # negloglikelihood = F.nll_loss(F.log_softmax(self(x), dim=1), y.data)
            self.zero_grad()
            loglikelihoods.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_estimated_mean{}'.format(n, 9 if self.online else task_id),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and task_id>= 1:
                    existing_values = getattr(self, '{}_estimated_fisher9'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(
                    '{}_estimated_fisher{}'.format(n, 9 if self.online else task_id),
                    est_fisher_info[n])
        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self, task_id=0, lamda=1e9):
        if task_id > 0:
            losses = []
            for task in range(0, task_id):
                for n, p in self.named_parameters():
                    # retrieve the consolidated mean and fisher information.
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_estimated_mean{}'.format(n,9 if self.online else task))
                        fisher = getattr(self, '{}_estimated_fisher{}'.format(n,9 if self.online else task))
                        # wrap mean and fisher in Vs.
                        # mean = V(mean)
                        fisher = self.gamma*fisher if self.online else fisher
                        # calculate a ewc loss. (assumes the parameter's prior as
                        # gaussian distribution with the estimated mean and the
                        # estimated cramer-rao lower bound variance, which is
                        # equivalent to the inverse of fisher information)
                        losses.append((fisher * (p-mean)**2).sum())
            return (lamda/2)*sum(losses)
        else:
            return torch.tensor(0., device=self.device)
