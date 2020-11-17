import torch
import copy
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from gan_training.utils_model_load import howMny_componetes, chanel_percent, my_copy
import torch.utils.data.distributed
F_conv = torch.nn.functional.conv2d
from torch.distributions.gamma import Gamma
gammaa = Gamma(torch.tensor([0.1]), torch.tensor([1.0]))
S_rep = 'abs'
S_scale_g, S_scale_b = 100.0, 20.0

stdd1, stdd2, stdd3, stdd4 = 1e-1, 1e-2, 1e-3, 1e-4
is_use_bias = True

class AdaFM_fc(nn.Module):
    def __init__(self, in_channel, only_para=False):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(1, in_channel))
        self.beta = nn.Parameter(torch.zeros(1, in_channel))
        self.global_gamma = []
        self.global_beta = []

        # if is_use_bias:
            # self.b = nn.Parameter(torch.zeros(1, in_channel))
            # self.global_b = []

    def forward(self, input=0, W=0, b=0, b_i=0, task_id=-1, UPDATE_GLOBAL=False):

        if not UPDATE_GLOBAL:
            if task_id >= 0: # test history tasks
                W0 = W.t() * self.global_gamma[task_id] + self.global_beta[task_id]
                out = input.mm(W0) + b
                if is_use_bias:
                    # out += self.global_b[task_id]
                    out += b_i

            elif task_id == -1: # train the current task
                W0 = W.t() * self.gamma + self.beta
                out = input.mm(W0) + b + b_i
            return out
        else:
            self.global_gamma.append(my_copy(self.gamma))
            self.global_beta.append(my_copy(self.beta))
            # if is_use_bias:
            #     self.global_b.append(my_copy(b_i.clone()))



class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.nlabels = nlabels
        embed_size = nlabels
        self.z_dim = z_dim
        self.global_AdaFM_class_bias = []

        # Submodules
        self.embedding = nn.Embedding(nlabels, 1)
        self.fc = nn.Linear(z_dim + 1, 16*nf*s0*s0)
        self.AdaFM_class_bias = nn.Linear(embed_size, 16*nf*s0*s0)
        # self.AdaFM_fc_b = nn.Parameter(torch.zeros(16*nf*s0*s0))
        self.AdaFM_fc = AdaFM_fc(16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock_style_SVD(16*nf, 16*nf)
        self.resnet_1_0 = ResnetBlock_style_SVD(16*nf, 16*nf)
        self.resnet_2_0 = ResnetBlock_style_SVD(16*nf, 8*nf)
        self.resnet_3_0 = ResnetBlock_style_SVD(8*nf, 4*nf)
        self.resnet_4_0 = ResnetBlock_style_SVD(4*nf, 2*nf)
        self.resnet_5_0 = ResnetBlock_style_SVD(2*nf, 1*nf)
        self.resnet_6_0 = ResnetBlock_style_SVD(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 7, padding=3)
        # self.AdaFM_conv = AdaFM(3, nf)

    def forward(self, z=0, y=0, task_id=-1, UPDATE_GLOBAL=False, TH=1e-3, FRAC0=0.5, FRAC1=0.9, n_ch=256, Iterr=1, is_FID=False):
        # assert(z.size(0) == y.size(0))
        if not UPDATE_GLOBAL:
            batch_size = z.size(0)
            yembed1, yembed2 = my_embedding(y, nlabels=self.nlabels)
            yz = torch.cat([z, yembed1], dim=1)
            # out = self.fc(yz)
            W_fc = self.fc.weight
            b_fc = self.fc.bias
            if task_id == -1:
                b_i = self.AdaFM_class_bias(yembed2)
            else:
                b_i = self.global_AdaFM_class_bias[task_id](yembed2)
            out = self.AdaFM_fc(yz, W=W_fc, b=b_fc, b_i=b_i,task_id=task_id)
            out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

            out = self.resnet_0_0(out, task_id=task_id)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_1_0(out, task_id=task_id)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_2_0(out, task_id=task_id)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_3_0(out, task_id=task_id)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_4_0(out, task_id=task_id)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_5_0(out, task_id=task_id)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_6_0(out, task_id=task_id)

            out = self.conv_img(actvn(out))
            out = torch.tanh(out)
            if is_FID:
                out = F.interpolate(out, 128, mode='bilinear')
            return out, batch_size
        else:
            out = z
            self.AdaFM_fc(out, task_id=task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.global_AdaFM_class_bias.append(copy.deepcopy(self.AdaFM_class_bias))
            self.resnet_0_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_1_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_2_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_3_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_4_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_5_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_6_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)



class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter

        # Submodules
        self.conv_img = nn.Conv2d(3, 1*nf, 7, padding=3)

        self.resnet_0_0 = ResnetBlock_style(1 * nf, 1 * nf)
        self.resnet_1_0 = ResnetBlock_style(1 * nf, 2 * nf)
        self.resnet_2_0 = ResnetBlock_style(2 * nf, 4 * nf)
        self.resnet_3_0 = ResnetBlock_style(4 * nf, 8 * nf)
        self.resnet_4_0 = ResnetBlock_style(8 * nf, 16 * nf)
        self.resnet_5_0 = ResnetBlock_style(16 * nf, 16 * nf)
        self.resnet_6_0 = ResnetBlock_style(16 * nf, 16 * nf)

        self.fc = nn.Linear(16*nf*s0*s0, nlabels)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet_0_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_2_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_4_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_5_0(out)

        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_6_0(out)

        out = out.view(batch_size, 16*self.nf*self.s0*self.s0)
        out = self.fc(actvn(out))

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class ResnetBlock_style(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.lrelu_0 = nn.LeakyReLU(0.2)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.AdaFM_0 = AdaFM(self.fhidden, self.fin)
        self.AdaFM_b0 = nn.Parameter(torch.zeros(self.fhidden))

        self.lrelu_1 = nn.LeakyReLU(0.2)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        self.AdaFM_1 = AdaFM(self.fout, self.fhidden)
        self.AdaFM_b1 = nn.Parameter(torch.zeros(self.fout))

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)

        dx = self.lrelu_0(x)
        dx = self.AdaFM_0(dx, self.conv_0.weight, self.conv_0.bias, self.AdaFM_b0)

        dx = self.lrelu_1(dx)
        dx = self.AdaFM_1(dx, self.conv_1.weight, self.conv_1.bias, self.AdaFM_b1)

        out = x_s + 0.1 * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class AdaFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.style_gama = nn.Parameter(torch.ones(in_channel, out_channel, 1, 1))
        self.style_beta = nn.Parameter(torch.zeros(in_channel, out_channel, 1, 1))
        

    def forward(self, input, W, b, b_i=0):
        W_i = W * self.style_gama + self.style_beta
        if is_use_bias:
            b_i = b_i
        out = F_conv(input, W_i, bias=b + b_i, stride=1, padding=1)
        return out


class AdaFM_SVD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.global_num_task = 0
        self.global_gamma = []
        self.global_beta = []

        self.style_gama = nn.Parameter(torch.ones(in_channel, out_channel, 1, 1))
        self.style_beta = nn.Parameter(torch.zeros(in_channel, out_channel, 1, 1))
        if is_use_bias:
            # self.b = nn.Parameter(torch.zeros(in_channel))
            self.global_b = []

    def forward(self, input=0, W=0, b=0, b_i=0.0, task_id=-1, Iterr=1):

        if task_id >= 0: # test history tasks
            if is_use_bias:
                b_i = self.global_b[task_id]
            gamma = self.global_gamma[task_id]
            beta = self.global_beta[task_id]
        elif task_id == -1:
            if is_use_bias:
                # b_i = self.b
                b_i = b_i
            gamma = self.style_gama.squeeze()
            beta = self.style_beta.squeeze()

        W_i = gamma.unsqueeze(2).unsqueeze(3) * W + beta.unsqueeze(2).unsqueeze(3)
        out = F_conv(input, W_i, bias=b + b_i, stride=1, padding=1)
        return out

    def update_global(self, b_i=0):
        if is_use_bias:
            self.global_b.append(my_copy(b_i))

        if self.global_num_task >= 0:
            with torch.no_grad():
                self.global_gamma.append(my_copy(self.style_gama.squeeze()))
                self.global_beta.append(my_copy(self.style_beta.squeeze()))
                self.global_num_task += 1



def Para2List(L_P):
    n = L_P.__len__()
    L_t = []
    s_sum = 0.0
    for i in range(n):
        L_t.append(my_copy(L_P[i]))
        s_sum += L_P[i].sum().detach().data * 1.0
    return L_t, s_sum


class ResnetBlock_style_SVD(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.AdaFM_0 = AdaFM_SVD(self.fhidden, self.fin)
        self.AdaFM_b0 = nn.Parameter(torch.zeros(self.fhidden))

        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        self.AdaFM_1 = AdaFM_SVD(self.fout, self.fhidden)
        self.AdaFM_b1 = nn.Parameter(torch.zeros(self.fout))

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, task_id=-1, UPDATE_GLOBAL=False):
        if not UPDATE_GLOBAL:
            x_s = self._shortcut(x)
            dx = actvn(x)
            dx = self.AdaFM_0(dx, self.conv_0.weight, self.conv_0.bias, b_i=self.AdaFM_b0, task_id=task_id)

            dx = actvn(dx)
            dx = self.AdaFM_1(dx, self.conv_1.weight, self.conv_1.bias, b_i=self.AdaFM_b1, task_id=task_id)

            out = x_s + 0.1 * dx

            return out
        else:
            self.AdaFM_0.update_global(self.AdaFM_b0)
            self.AdaFM_1.update_global(self.AdaFM_b1)

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class batch_instance_norm(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bn_ = nn.BatchNorm2d(channel, affine=False)
        self.in_ = nn.InstanceNorm2d(channel, affine=False)
        self.gamma = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel, 1, 1))

        self.rho = nn.Parameter(torch.ones(1, channel, 1, 1))

    def forward(self, input):

        h_bn = self.bn_(input)
        h_in = self.in_(input)

        h = self.rho * h_bn + (1 - self.rho) * h_in
        h = h * self.gamma + self.beta
        return h


def norm_u(u):
    TH = torch.tensor(1e-5, device=u.device)
    norm_ = torch.norm(u, dim=0, keepdim=True)
    out = u / (norm_+TH)
    return out

def my_copy(x):
    x_copy = x.detach().data * 1.0
    return x_copy


def my_embedding(y, nlabels=1):
    e_y=torch.zeros(y.shape[0], nlabels+1,device=y.device).scatter_(1, (y+1).unsqueeze(1), 1)
    # print('e_y.shape===============', e_y.shape)
    y0 = 0.8393
    for ii in range(y.shape[0]):
        e_y[ii,0] = y0
    # print('e_y.shape===============', e_y)
    ey1 = e_y[:,0].unsqueeze(1)
    ey2 = e_y[:, 1:]
    return ey1, ey2