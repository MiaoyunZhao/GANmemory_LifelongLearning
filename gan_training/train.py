# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd


class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer,
                 gan_type, reg_type, reg_param, data_fix='ImageNet', CONS_lowrank=1e-2):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.data_fix = data_fix
        
        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.CONS_lowrank=CONS_lowrank

    def generator_trainstep(self, y, z, Iterr=0, is_lowrank=False):
        assert(y.size(0) == z.size(0))
        self.generator.train()
        self.discriminator.train()
        self.g_optimizer.zero_grad()

        x_fake, loss_w = self.generator(z, y, Iterr=Iterr)
        d_fake = self.discriminator(x_fake, y)
        gloss = self.compute_loss(d_fake, 1)

        # ----------- low rank loss -----------
        if is_lowrank and Iterr%10==0:
            lr_loss = self.lowrank_constr(z.device, lamda_gama=self.CONS_lowrank, lamda_beta=self.CONS_lowrank)
            (gloss + lr_loss).backward()
        else:
            gloss.backward()
            lr_loss = gloss

        self.g_optimizer.step()
        return gloss.item(), x_fake.detach(), lr_loss.item()

    def discriminator_trainstep(self, x_real, y, x_fake0, Iterr=0):
        self.generator.train()
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()

        d_real = self.discriminator(x_real, y)
        dloss_real = self.compute_loss(d_real, 1)

        if Iterr%2==0 and (self.reg_type == 'real' or self.reg_type == 'real_fake'):
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real, x_real).mean()
            reg.backward()
        else:
            dloss_real.backward()
            reg = dloss_real

        x_fake0.requires_grad_()
        d_fake = self.discriminator(x_fake0, y)
        dloss_fake = self.compute_loss(d_fake, 0)

        if Iterr%2==0 and (self.reg_type == 'fake' or self.reg_type == 'real_fake'):
            dloss_fake.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_fake, x_fake0).mean()
            reg.backward()
        else:
            dloss_fake.backward()

        if self.reg_type == 'wgangp':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake0, y)
            reg.backward()
        elif self.reg_type == 'wgangp0':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake0, y, center=0.)
            reg.backward()

        self.d_optimizer.step()

        # Output
        dloss = (dloss_real + dloss_fake)

        if self.reg_type == 'none':
            reg = torch.tensor(0.)

        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2*target - 1) * d_out.mean()

        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

    def lowrank_constr_v2(self, lamda_gama=1e-3, lamda_beta=1e-3):
        rank_gama, rank_beta=0., 0.
        for k, v in self.generator.named_parameters():
            if k.find('style_gama') >= 0:
                rank_gama += v.squeeze().norm(p='nuc')
            if k.find('style_beta') >= 0:
                rank_beta += v.squeeze().norm(p='nuc')
        reg = lamda_gama * rank_gama + lamda_beta * rank_beta
        return reg

    def lowrank_constr(self, device, lamda_gama=1e-3, lamda_beta=1e-3):
        rank_gama, rank_beta=0., 0.
        for k, v in self.generator.named_parameters():
            if k.find('style_gama') >= 0:
                _, s, _ = v.squeeze().svd()
                ind = torch.tensor(range(s.shape[0]), device=device).float()
                ind_embed = 0.1 + ((10.0 * ind / s.shape[0]).sigmoid())
                rank_gama += (s * ind_embed).abs().sum()
            if k.find('style_beta') >= 0:
                _, s, _ = v.squeeze().svd()
                ind = torch.tensor(range(s.shape[0]), device=device).float()
                ind_embed = 0.1 + ((10.0 * ind / s.shape[0]).sigmoid())
                rank_beta += (s * ind_embed).abs().sum()
        reg = lamda_gama * rank_gama + lamda_beta * rank_beta
        return reg


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    # toggle_grad(model_src, False)
    # toggle_grad(model_tgt, False)
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
