import torch
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
S_scale_g, S_scale_b = 1.0, 1.

stdd1, stdd2, stdd3, stdd4 = 1e-1, 1e-2, 1e-3, 1e-4
is_use_bias = True


class AdaFM_fc(nn.Module):
    def __init__(self, in_channel, only_para=False):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(1, in_channel))
        self.beta = nn.Parameter(torch.zeros(1, in_channel))
        self.global_gamma = []
        self.global_beta = []

        if is_use_bias:
            self.global_b = []
            self.b = nn.Parameter(torch.zeros(in_channel))

    def forward(self, input=0, W=0, b=0, b_i=0, task_id=-1, UPDATE_GLOBAL=False):

        if not UPDATE_GLOBAL:
            if task_id >= 0: # test history tasks
                W0 = W.t() * self.global_gamma[task_id] + self.global_beta[task_id]
                out = input.mm(W0) + b
                if is_use_bias:
                    out += self.global_b[task_id]

            elif task_id == -1: # train the current task
                W0 = W.t() * self.gamma + self.beta
                out = input.mm(W0) + b
                if is_use_bias:
                      out = out + self.b
            return out
        else:

            self.global_gamma.append(my_copy(self.gamma))
            self.global_beta.append(my_copy(self.beta))
            if is_use_bias:
                self.global_b.append(my_copy(self.b))


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=1, nfilter=64, **kwargs):
        super().__init__()
        s0 = self.s0 = size // 64
        nf = self.nf = nfilter
        self.z_dim = z_dim

        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim+embed_size, 16*nf*s0*s0)
        self.AdaFM_fc = AdaFM_fc(16*nf*s0*s0)

        self.resnet_0_0 = ResnetBlock_style_SVD(16*nf, 16*nf)
        self.resnet_1_0 = ResnetBlock_style_SVD(16*nf, 16*nf)
        self.resnet_2_0 = ResnetBlock_style_SVD(16*nf, 8*nf)
        self.resnet_3_0 = ResnetBlock_style_SVD(8*nf, 4*nf)
        self.resnet_4_0 = ResnetBlock_style(4*nf, 2*nf)
        self.resnet_5_0 = ResnetBlock_style(2*nf, 1*nf)
        self.resnet_6_0 = ResnetBlock_style(1*nf, 1*nf)

        self.conv_img = nn.Conv2d(nf, 3, 7, padding=3)
        # self.AdaFM_conv = AdaFM(3, nf)

    def forward(self, z=0, y=0, task_id=-1, UPDATE_GLOBAL=False, TH=1e-3, FRAC0=0.5, FRAC1=0.9, n_ch=256, Iterr=1, is_FID=False,device=None):
        # assert(z.size(0) == y.size(0))
        batch_size = z.size(0)
        if not UPDATE_GLOBAL:
            yembed = self.embedding(y)
            yz = torch.cat([z, yembed], dim=1)
            W_fc = self.fc.weight
            b_fc = self.fc.bias
            out = self.AdaFM_fc(yz, W=W_fc, b=b_fc, task_id=task_id)
            out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

            out = self.resnet_0_0(out, task_id=task_id, Iterr=Iterr)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_1_0(out, task_id=task_id, Iterr=Iterr)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_2_0(out, task_id=task_id, Iterr=Iterr)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_3_0(out, task_id=task_id, Iterr=Iterr)

            out = F.interpolate(out, scale_factor=2)
            # out = self.resnet_4_0(out, task_id=task_id, Iterr=Iterr)
            out = self.resnet_4_0(out)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_5_0(out)

            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_6_0(out)

            out = self.conv_img(actvn(out))
            out = torch.tanh(out)
            if is_FID:
                out = F.interpolate(out, 128, mode='bilinear')
            return out, batch_size
        else:
            out = z
            self.AdaFM_fc(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.resnet_0_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            self.resnet_1_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            self.resnet_2_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            self.resnet_3_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            # self.resnet_4_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            # self.resnet_5_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            # self.resnet_6_0(out, task_id, UPDATE_GLOBAL=UPDATE_GLOBAL, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)



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

        self.lrelu_1 = nn.LeakyReLU(0.2)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        self.AdaFM_1 = AdaFM(self.fout, self.fhidden)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)

        dx = self.lrelu_0(x)
        dx = self.AdaFM_0(dx, self.conv_0.weight, self.conv_0.bias)

        dx = self.lrelu_1(dx)
        dx = self.AdaFM_1(dx, self.conv_1.weight, self.conv_1.bias)

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
        if is_use_bias:
            self.b = nn.Parameter(torch.zeros(in_channel))

    def forward(self, input, W, b, b_i=0):
        W_i = W * self.style_gama + self.style_beta
        if is_use_bias:
            b_i = 1.0 * self.b
        out = F_conv(input, W_i, bias=b + b_i, stride=1, padding=1)
        return out


class AdaFM_multiTask(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.style_gama = nn.Parameter(torch.ones(in_channel, out_channel, 1, 1))
        self.style_beta = nn.Parameter(torch.zeros(in_channel, out_channel, 1, 1))
        self.global_gamma = []
        self.global_beta = []

        if is_use_bias:
            self.global_b = []
            self.b = nn.Parameter(torch.zeros(in_channel))

    def forward(self, input, W, b, b_i=0, task_id=-1, UPDATE_GLOBAL=False):
        if not UPDATE_GLOBAL:
            if task_id >= 0: # test history tasks
                W_i = W * self.global_gamma[task_id] + self.global_beta[task_id]
                if is_use_bias:
                    b_i = self.global_b[task_id]
                out = F_conv(input, W_i, bias=b + b_i, stride=1, padding=1)

            elif task_id == -1: # train the current task
                W_i = W * self.style_gama + self.style_beta
                if is_use_bias:
                    b_i = self.b
                out = F_conv(input, W_i, bias=b + b_i, stride=1, padding=1)
            return out
        else:
            self.global_gamma.append(my_copy(self.style_gama))
            self.global_beta.append(my_copy(self.style_beta))
            if is_use_bias:
                self.global_b.append(my_copy(self.b))


class ResnetBlock_style_multiTask(nn.Module):
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
        self.AdaFM_0 = AdaFM_multiTask(self.fhidden, self.fin)

        self.lrelu_1 = nn.LeakyReLU(0.2)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        self.AdaFM_1 = AdaFM_multiTask(self.fout, self.fhidden)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, task_id=-1, UPDATE_GLOBAL=False, Iterr=1, device=None):
        if not UPDATE_GLOBAL:
            x_s = self._shortcut(x)
            dx = self.lrelu_0(x)
            dx = self.AdaFM_0(dx, self.conv_0.weight, self.conv_0.bias, task_id=task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            dx = self.lrelu_1(dx)
            dx = self.AdaFM_1(dx, self.conv_1.weight, self.conv_1.bias, task_id=task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)

            out = x_s + 0.1 * dx

            return out
        else:
            self.AdaFM_0.update_global(self.fhidden, self.fin, task_id=task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)
            self.AdaFM_1.update_global(self.fout, self.fhidden, task_id=task_id, UPDATE_GLOBAL=UPDATE_GLOBAL)

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class AdaFM_SVD(nn.Module):
    def __init__(self, in_channel, out_channel, global_num_gamma=0, global_num_beta=0, global_num_task=0):
        super().__init__()

        n_task = 10

        self.global_num_gamma = []
        self.global_num_beta = []
        self.global_num_task = 0

        self.global_gamma_u = []
        self.global_gamma_s = []
        self.global_gamma_v = []

        self.global_beta_u = []
        self.global_beta_s = []
        self.global_beta_v = []

        self.style_gama = nn.Parameter(stdd2 + stdd2 * torch.randn(in_channel, out_channel,1,1))
        self.style_beta = nn.Parameter(stdd2 + stdd2 * torch.randn(in_channel, out_channel,1,1))
        if is_use_bias:
            self.b = nn.Parameter(torch.zeros(in_channel))
            self.global_b = []

    def forward(self, input=0, W=0, b=0, b_i=0.0, task_id=-1, Iterr=1):
        if task_id >= 0: # test history tasks
            # print('task_id ================ ', task_id)
            if is_use_bias:
                b_i = self.global_b[task_id]
                # print('self.global_b[0] 0================ ', b_i)

            ss = self.global_gamma_s[task_id]
            uu = list_2_matrix(self.global_gamma_u[:task_id+1], global_num=self.global_num_gamma)
            vv = list_2_matrix(self.global_gamma_v[:task_id+1], global_num=self.global_num_gamma)
            gamma = (uu * ss.unsqueeze(0)).mm(vv.t())

            ss = self.global_beta_s[task_id]
            uu = list_2_matrix(self.global_beta_u[:task_id+1], global_num=self.global_num_beta)
            vv = list_2_matrix(self.global_beta_v[:task_id+1], global_num=self.global_num_beta)
            beta = (uu * ss.unsqueeze(0)).mm(vv.t())

        elif task_id == -1:
            if is_use_bias:
                b_i = self.b
                # print('self.global_b[0] -1================ ', b_i)
                # b_i = self.global_b[0]
            if self.global_num_task > 0:
                ss = self.gamma_s1
                uu = list_2_matrix(self.global_gamma_u, global_num=self.global_num_gamma)
                vv = list_2_matrix(self.global_gamma_v, global_num=self.global_num_gamma)
                gamma = (uu * ss.unsqueeze(0)).mm(vv.t())
                ss = self.beta_s1
                uu = list_2_matrix(self.global_beta_u, global_num=self.global_num_beta)
                vv = list_2_matrix(self.global_beta_v, global_num=self.global_num_beta)
                beta = (uu * ss.unsqueeze(0)).mm(vv.t())
            else:
                gamma = 0.
                beta = 0.
            gamma += self.style_gama.squeeze()
            beta += self.style_beta.squeeze()
        # print('gamma.shape=============================', gamma.shape)
        # print('gamma.shape=============================', beta.shape)

        W_i = gamma.to(input.device).unsqueeze(2).unsqueeze(3) * W.to(input.device) + beta.to(input.device).unsqueeze(2).unsqueeze(3)
        out = F_conv(input, W_i, bias=b + b_i, stride=1, padding=1)
        return out

    def update_global(self, in_channel, out_channel, task_id=0, FRAC0=0.5, FRAC1=0.9, n_ch=256,device=None):
        # FRAC0, FRAC1 = 0.5, 0.99
        p = [1.0, 1.0, 0.95, 0.9, 0.8]
        p2 = [1.0, 1.0, 0.95, 0.9, 0.8]
        p2 = [1.0, 1.0, 0.95, 0.9, 0.8]
        p4 = [1.0, 1.0, 0.95, 0.9, 0.8]
        if is_use_bias:
            self.global_b.append(my_copy(self.b))

        if self.global_num_task >= 0:
            with torch.no_grad():
                # gamma
                #
                if self.global_num_task == 0:
                    s_sum = 0.0
                    FRAC = chanel_percent(self.style_gama.shape[0], p=p)
                elif self.global_num_task >= 4:
                    s_sum = self.gamma_s1.abs().sum()
                    FRAC = chanel_percent(self.style_gama.shape[0], p=p4)
                else:
                    s_sum = self.gamma_s1.abs().sum()
                    FRAC = chanel_percent(self.style_gama.shape[0], p=p2) #[0.95, 0.95, 0.9, 0.9, 0.9]
                # FRAC = chanel_percent(self.style_gama.shape[0])
                Ua, Sa, Va = self.style_gama.squeeze().svd()
                ii, jj = Sa.abs().sort(descending=True)
                ii[0] += s_sum
                ii_acsum = ii.cumsum(dim=0)
                if s_sum / ii_acsum[-1] < FRAC:
                    num = (~(ii_acsum / ii_acsum[-1] >= FRAC)).sum() + 1
                    if self.global_num_task == 0:
                        s_all = my_copy(Sa[jj[:num]])
                    else:
                        s_all = torch.cat((my_copy(self.gamma_s1), my_copy(Sa[jj[:num]])), 0)

                    self.global_gamma_s.append(s_all)
                    self.global_gamma_u.append(my_copy(Ua[:, jj[:num]]))
                    self.global_gamma_v.append(my_copy(Va[:, jj[:num]]))
                    self.global_num_gamma.append(num)
                else:
                    num = jj[0]-jj[0]
                    self.global_num_gamma.append(num)
                    self.global_gamma_s.append(my_copy(self.gamma_s1))
                    self.global_gamma_u.append('none')
                    self.global_gamma_v.append('none')
                # beta
                if self.global_num_task == 0:
                    s_sum = 0.0
                    FRAC = chanel_percent(self.style_beta.shape[0], p=p)
                elif self.global_num_task >= 4:
                    s_sum = self.beta_s1.abs().sum()
                    FRAC = chanel_percent(self.style_gama.shape[0], p=p4)
                else:
                    s_sum = self.beta_s1.abs().sum()
                    FRAC = chanel_percent(self.style_gama.shape[0], p=p2) #[0.95, 0.95, 0.9, 0.9, 0.9]
                # FRAC = chanel_percent(self.style_beta.shape[0])
                Ua, Sa, Va = self.style_beta.squeeze().svd()
                ii, jj = Sa.abs().sort(descending=True)
                ii[0] += s_sum
                ii_acsum = ii.cumsum(dim=0)
                if s_sum / ii_acsum[-1] < FRAC:
                    num = (~(ii_acsum / ii_acsum[-1] >= FRAC)).sum() + 1
                    if self.global_num_task == 0:
                        s_all = my_copy(Sa[jj[:num]])
                    else:
                        s_all = torch.cat((my_copy(self.beta_s1), my_copy(Sa[jj[:num]])), 0)

                    self.global_num_task = self.global_num_task + 1
                    self.global_beta_s.append(s_all)
                    self.global_beta_u.append(my_copy(Ua[:, jj[:num]]))
                    self.global_beta_v.append(my_copy(Va[:, jj[:num]]))
                    self.global_num_beta.append(num)
                else:
                    num = jj[0]-jj[0]
                    self.global_num_beta.append(num)
                    self.global_beta_s.append(my_copy(self.beta_s1))
                    self.global_beta_u.append('none')
                    self.global_beta_v.append('none')
            # update parameters
            # self.gamma_s1 = nn.Parameter(self.global_gamma_s[0][0])
            # for ii in range(1, self.global_num_task):
            #     self.gamma_s1.append(nn.Parameter(stdd3 * torch.rand(self.global_num_gamma[ii], device=device)))
            self.gamma_s1 = nn.Parameter(stdd4*torch.randn(sum(self.global_num_gamma), device=device))
            # self.gamma_s1.data[:self.global_num_gamma[0]] = self.global_gamma_s[0].data
            self.gamma_s1.data[:sum(self.global_num_gamma)] = self.global_gamma_s[-1].data

            # self.beta_s1 = nn.Parameter(self.global_beta_s[0][0])
            # for ii in range(1, self.global_num_task):
            #     self.beta_s1.append(nn.Parameter(stdd3 * torch.rand(self.global_num_beta[ii], device=device)))
            self.beta_s1 = nn.Parameter(stdd4*torch.randn(sum(self.global_num_beta), device=device))
            # self.beta_s1.data[:self.global_num_beta[0]] = self.global_beta_s[0].data
            self.beta_s1.data[:sum(self.global_num_beta)] = self.global_beta_s[-1].data

        # self.gamma = nn.Parameter(1/(in_channel**0.5) * torch.randn(in_channel, out_channel))
        # self.beta = nn.Parameter(stdd2 + stdd2 * torch.randn(in_channel, out_channel))
        self.style_gama.data = torch.zeros(in_channel, out_channel).to(device)
        self.style_beta.data = torch.zeros(in_channel, out_channel).to(device)
        if is_use_bias:
            # self.b = nn.Parameter(torch.zeros(in_channel))
            self.b.data = my_copy(torch.tensor(self.global_b[0]).clone()).to(device)
        # print('style_gama device =================== ', self.style_gama.device, self.style_beta.device)



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

        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        self.AdaFM_1 = AdaFM_SVD(self.fout, self.fhidden)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x, task_id=-1, UPDATE_GLOBAL=False, FRAC0=0.5, FRAC1=0.9, n_ch=256, Iterr=1, device=None):
        if not UPDATE_GLOBAL:
            x_s = self._shortcut(x)
            dx = actvn(x)
            dx = self.AdaFM_0(dx, self.conv_0.weight, self.conv_0.bias, task_id=task_id, Iterr=Iterr)

            dx = actvn(dx)
            dx = self.AdaFM_1(dx, self.conv_1.weight, self.conv_1.bias, task_id=task_id, Iterr=Iterr)

            out = x_s + 0.1 * dx

            return out
        else:
            self.AdaFM_0.update_global(self.fhidden, self.fin, task_id=task_id, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)
            self.AdaFM_1.update_global(self.fout, self.fhidden, task_id=task_id, FRAC0=FRAC0, FRAC1=FRAC1, n_ch=n_ch,device=device)

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


def list_2_matrix(x, global_num=0):
    N = x.__len__()
    output = x[0]
    for ii in range(1, N):
        if global_num[ii] > 0:
            output = torch.cat((output, x[ii]), 1)
    return output
