
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total=', total_num, 'Trainable=', trainable_num, 'fixed=', total_num-trainable_num)


def load_part_model(m_fix, m_ini):
    dict_fix = m_fix.state_dic()
    dict_ini = m_ini.state_dic()

    dict_fix = {k: v for k, v in dict_fix.items() if k in dict_ini and k.find('embedding')==-1 and k.find('fc') == -1}
    dict_ini.update(dict_fix)
    m_ini.load_state_dict(dict_ini)
    return m_ini



def model_equal_all(model, dict):
    model_dict = model.state_dict()
    model_dict.update(dict)
    model.load_state_dict(model_dict)
    return model


def change_model_name(model, pretrained_net_dict):
    # pretrained_net_dict = dict
    new_state_dict = OrderedDict()
    for k, v in pretrained_net_dict.items():
        if k.find('AdaFM') >= 0 and k.find('style_gama') >= 0:
            indd = k.find('style_gama')
            name = k[:indd]+'gamma'
            new_state_dict[name] = v.squeeze()
        elif k.find('AdaFM') >= 0 and k.find('style_beta') >= 0:
            indd = k.find('style_beta')
            name = k[:indd]+'beta'
            new_state_dict[name] = v.squeeze()
        else:
            new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def save_adafm_only(model, is_G=True):
    new_state_dict = OrderedDict()
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if k.find('AdaFM') >= 0:
            name = k
            new_state_dict[name] = v
        if is_G==False:
            if k.find('fc') >= 0:
                name = k
                new_state_dict[name] = v

    return new_state_dict


def model_equal_part(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1}
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model


def model_equal_part_embed(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if k in model_dict and k.find('embedding') == -1}
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model


def model_equal_embeding(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1}
    model_dict.update(dict_fix)
    for k, v in dict_all.items():
        if k.find('fc') >= 0 and k.find('weight') >=0:
            name = k
            model_dict[name][:,:257] = v
    model.load_state_dict(model_dict)
    return model



def model_load_interplation(generator, dict_G_1, dict_G_2, lamdd=0.0, block=None):
    model_dict = generator.state_dict()
    for k, v in dict_G_1.items():
        if block == 9:
            model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
        elif block==0:
            if k.find('resnet_0_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]
        elif block==1:
            if k.find('resnet_1_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]
        elif block==2:
            if k.find('resnet_2_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]
        elif block==3:
            if k.find('resnet_3_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]
        elif block==4:
            if k.find('resnet_4_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]
        elif block==5:
            if k.find('resnet_5_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]
        elif block==6:
            if k.find('resnet_6_0')>=0:
                model_dict[k] = (1-lamdd)*dict_G_1[k] + lamdd*dict_G_2[k]
            else:
                model_dict[k] = dict_G_1[k]

    generator.load_state_dict(model_dict)
    return generator



def model_load_choose_para(generator, dict_G_1, para=None):
    model_dict = generator.state_dict()
    for k, v in dict_G_1.items():
        if para == None:
            model_dict[k] = dict_G_1[k]
        elif para==0:
            if k.find('style_gama')>=0 or (k.find('AdaFM_fc.gamma')>=0):
                model_dict[k] = dict_G_1[k]
                print(k)
        elif para==1:
            if k.find('style_beta')>=0 or (k.find('AdaFM_fc.beta')>=0):
                model_dict[k] = dict_G_1[k]
                print(k)
        elif para==2:
            if k.find('AdaFM_b')>=0 or k.find('AdaFM_fc_b')>=0:
                model_dict[k] = dict_G_1[k]
                print(k)

    generator.load_state_dict(model_dict)
    return generator


def model_load_choose_layer(generator, dict_G_1, Layerr=None):
    model_dict = generator.state_dict()
    for k, v in dict_G_1.items():
        if Layerr == None:
            model_dict[k] = dict_G_1[k]
        else:
            if k.find(Layerr) >= 0 and (k.find('AdaFM') >= 0):
                model_dict[k] = dict_G_1[k]
                print(k)
    generator.load_state_dict(model_dict)
    return generator


def model_load_donot_choose_para(generator, dict_G_1, para=None):
    model_dict = generator.state_dict()
    for k, v in dict_G_1.items():
        if para == None:
            model_dict[k] = dict_G_1[k]
        elif para==0:
            if k.find('style_gama')==-1 and k.find('AdaFM_fc.gamma')==-1:
                model_dict[k] = dict_G_1[k]
        elif para==1:
            if k.find('style_beta')==-1 and k.find('AdaFM_fc.beta')==-1:
                model_dict[k] = dict_G_1[k]
        elif para==2:
            if k.find('AdaFM_b')==-1 and k.find('AdaFM_fc_b')==-1:
                model_dict[k] = dict_G_1[k]

    generator.load_state_dict(model_dict)
    return generator



def out_bias_to_in_bias(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if
                k in model_dict and k.find('AdaFM_fc_b') == -1 and k.find('AdaFM_b0') == -1 and k.find('AdaFM_b1') == -1}
    for k, v in dict_all.items():
        ind = k.find('AdaFM_fc_b')
        if ind >= 0:
            dict_fix[k[:ind] + 'AdaFM_fc.b'] = v
        ind = k.find('AdaFM_b0')
        if ind >= 0:
            dict_fix[k[:ind] + 'AdaFM_0.b'] = v
        ind = k.find('AdaFM_b1')
        if ind >= 0:
            dict_fix[k[:ind] + 'AdaFM_1.b'] = v
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model


def model_equal_classCondition(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1}
    model_dict.update(dict_fix)
    for k, v in dict_all.items():
        if k.find('fc') >= 0 and k.find('weight') >=0:
            name = k
            model_dict[name] = v * 0.0
            model_dict[name][:,:257] = v
    model.load_state_dict(model_dict)
    return model


def model_equal_CelebA(model, dict_all, dim_z=-1, dim_h=-1):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if
                k in model_dict and k.find('embedding') == -1}
    for k, v in dict_all.items():
        if k.find('fc') >=0 and k.find('weight') >=0:
            if dim_z >= 0 and dim_h >= 0:
                dict_fix[k] = v[:dim_h, :dim_z]
            elif dim_z >= 0:
                dict_fix[k] = v[:, :dim_z]
        if dim_h >= 0:
            if k.find('fc') >=0 and k.find('bias') >=0:
                dict_fix[k] = v[:dim_h]

    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model


def model_equal_SVD(model, dict_all, FRAC=0.9):
    model_dict = model.state_dict()
    # FRAC = 0.9
    for k, v in dict_all.items():
        if k.find('AdaFM') >= 0 and k.find('style_gama') >= 0:
            # print('shape of FC:', v.shape)
            Ua, Sa, Va = (v-1.).squeeze().svd()

            if Ua.shape[0] >= 512:
                FRAC = 0.6
            else:
                FRAC = 0.9

            ii, jj = Sa.abs().sort(descending=True)
            ii_acsum = ii.cumsum(dim=0)
            NUM = (1 - (ii_acsum / ii_acsum[-1] >= FRAC)).sum() + 1
            v_new = 1. + (Ua[:, :NUM] * Sa[:NUM].unsqueeze(0)).mm(Va[:, :NUM].t())
            dict_all[k] = v_new.unsqueeze(2).unsqueeze(3)


        elif k.find('AdaFM') >= 0 and k.find('style_beta') >= 0:
            Ua, Sa, Va = v.squeeze().svd()

            if Ua.shape[0] >= 512:
                FRAC = 0.6
            else:
                FRAC = 0.9

            ii, jj = Sa.abs().sort(descending=True)
            ii_acsum = ii.cumsum(dim=0)
            NUM = (1 - (ii_acsum / ii_acsum[-1] >= FRAC)).sum() + 1
            v_new = (Ua[:, :NUM] * Sa[:NUM].unsqueeze(0)).mm(Va[:, :NUM].t())
            dict_all[k] = v_new.unsqueeze(2).unsqueeze(3)

    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


def model_equal_SVD_v2(model, dict_all, task_id=-1, NUM=200, dim_z=-1):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if
                k in model_dict and k.find('AdaFM_0') == -1 and k.find('AdaFM_1') == -1}
    if dim_z >= 0:
        for k, v in dict_all.items():
            if k.find('fc') >= 0 and k.find('weight') >= 0:
                # print('shape of FC:', v.shape)
                dict_fix[k] = v[:, :dim_z]
    model_dict.update(dict_fix)

    pecen = 1./2.
    genh, S_rep = 2., 'abs'
    # genh, S_rep = 2., 'exp'
    for k, v in dict_all.items():
        ind = k.find('AdaFM_0')
        if ind >= 0 and k.find('style_gama') >= 0:
            # print('shape of FC:', v.shape)
            Ua, Sa, Va = v.squeeze().svd()
            model_dict[k[:ind + 7] + '.gamma_u'] = Ua[:, :NUM]
            model_dict[k[:ind + 7] + '.gamma_v'] = Va[:, :NUM]
            if task_id >= 1:
                if S_rep == 'abs':
                    model_dict[k[:ind + 7] + '.gamma_s2'] = Sa[:NUM] * pecen
                elif S_rep == 'x2':
                    model_dict[k[:ind + 7] + '.gamma_s2'] = (Sa[:NUM] * pecen).pow(1. / genh)
                elif S_rep == 'exp':
                    model_dict[k[:ind + 7] + '.gamma_s2'] = (Sa[:NUM] * pecen).log()
            else:
                model_dict[k[:ind + 7] + '.gamma_s2'] = Sa[:NUM]
        elif ind >= 0 and k.find('style_beta') >= 0:
            # print('shape of FC:', v.shape)
            Ua, Sa, Va = v.squeeze().svd()
            model_dict[k[:ind + 7] + '.beta_u'] = Ua[:, :NUM]
            model_dict[k[:ind + 7] + '.beta_v'] = Va[:, :NUM]
            if task_id >= 1:
                if S_rep == 'abs':
                    model_dict[k[:ind + 7] + '.beta_s2'] = Sa[:NUM] * pecen
                elif S_rep == 'x2':
                    model_dict[k[:ind + 7] + '.beta_s2'] = (Sa[:NUM] * pecen).pow(1. / genh)
                elif S_rep == 'exp':
                    model_dict[k[:ind + 7] + '.beta_s2'] = (Sa[:NUM] * pecen).log()
            else:
                model_dict[k[:ind + 7] + '.beta_s2'] = Sa[:NUM]
        ind = k.find('AdaFM_1')
        if ind >= 0 and k.find('style_gama') >= 0:
            # print('shape of FC:', v.shape)
            Ua, Sa, Va = v.squeeze().svd()
            model_dict[k[:ind + 7] + '.gamma_u'] = Ua[:, :NUM]
            model_dict[k[:ind + 7] + '.gamma_v'] = Va[:, :NUM]
            if task_id >= 1:
                if S_rep == 'abs':
                    model_dict[k[:ind + 7] + '.gamma_s2'] = Sa[:NUM] * pecen
                elif S_rep == 'x2':
                    model_dict[k[:ind + 7] + '.gamma_s2'] = (Sa[:NUM] * pecen).pow(1. / genh)
                elif S_rep == 'exp':
                    model_dict[k[:ind + 7] + '.gamma_s2'] = (Sa[:NUM] * pecen).log()
            else:
                model_dict[k[:ind + 7] + '.gamma_s2'] = Sa[:NUM]
        elif ind >= 0 and k.find('style_beta') >= 0:
            # print('shape of FC:', v.shape)
            Ua, Sa, Va = v.squeeze().svd()
            model_dict[k[:ind + 7] + '.beta_u'] = Ua[:, :NUM]
            model_dict[k[:ind + 7] + '.beta_v'] = Va[:, :NUM]
            if task_id >= 1:
                if S_rep == 'abs':
                    model_dict[k[:ind + 7] + '.beta_s2'] = Sa[:NUM] * pecen
                elif S_rep == 'x2':
                    model_dict[k[:ind + 7] + '.beta_s2'] = (Sa[:NUM] * pecen).pow(1. / genh)
                elif S_rep == 'exp':
                    model_dict[k[:ind + 7] + '.beta_s2'] = (Sa[:NUM] * pecen).log()
            else:
                model_dict[k[:ind + 7] + '.beta_s2'] = Sa[:NUM]

    model.load_state_dict(model_dict)

    return model



def model_equal_SVD_fenkai(model, dict_all, NUM1=100, NUM2=50):
    model_dict = model.state_dict()

    for k, v in dict_all.items():
        if k.find('AdaFM') >= 0 and k.find('style_gama') >= 0:
            Ua, Sa, Va = v.squeeze().svd()
            print('shape of FC:', NUM1)
            v_new = torch.mm(torch.mm(Ua[:, :NUM1], torch.diag(Sa[:NUM1])), Va[:, :NUM1].t())
            dict_all[k] = v_new.unsqueeze(2).unsqueeze(3)
        if k.find('AdaFM') >= 0 and k.find('style_beta') >= 0:
            Ua, Sa, Va = v.squeeze().svd()
            print('shape of FC:', NUM2)
            v_new = torch.mm(torch.mm(Ua[:, :NUM2], torch.diag(Sa[:NUM2])), Va[:, :NUM2].t())
            dict_all[k] = v_new.unsqueeze(2).unsqueeze(3)

    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])



def svd_all_layers(dict_G, FRAC=0.9):
    flower_gamma_U, flower_gamma_S, flower_gamma_V = [], [], []
    flower_beta_U, flower_beta_S, flower_beta_V = [], [], []
    for k, v in dict_G.items():
        if k.find('AdaFM') >= 0 and k.find('style_gama') >= 0:
            # print('shape of FC:', v.shape)
            Ua, Sa, Va = (v - 1.).squeeze().svd()

            ii, jj = Sa.abs().sort(descending=True)
            ii_acsum = ii.cumsum(dim=0)
            NUM = (1 - (ii_acsum / ii_acsum[-1] >= FRAC)).sum() + 1

            flower_gamma_U.append(Ua[:, :NUM])
            flower_gamma_S.append(Sa[:NUM])
            flower_gamma_V.append(Va[:, :NUM])

        elif k.find('AdaFM') >= 0 and k.find('style_beta') >= 0:
            Ua, Sa, Va = v.squeeze().svd()

            ii, jj = Sa.abs().sort(descending=True)
            ii_acsum = ii.cumsum(dim=0)
            NUM = (1 - (ii_acsum / ii_acsum[-1] >= FRAC)).sum() + 1

            flower_beta_U.append(Ua[:, :NUM])
            flower_beta_S.append(Sa[:NUM])
            flower_beta_V.append(Va[:, :NUM])

    return flower_gamma_U, flower_gamma_S, flower_gamma_V, flower_beta_U, flower_beta_S, flower_beta_V


def gamma_beta_all_layers(dict_G):
    flower_gamma = []
    flower_beta = []
    for k, v in dict_G.items():
        if k.find('AdaFM') >= 0 and k.find('style_gama') >= 0:
            # print('shape of FC:', v.shape)
            flower_gamma.append(v.squeeze())

        elif k.find('AdaFM') >= 0 and k.find('style_beta') >= 0:
            flower_beta.append(v.squeeze())

    return flower_gamma, flower_beta


def cumpute_atc_num(generator_test, para='gamma', task=0, num_task=6):
    num_task = num_task - 1
    if para == 'gamma':
        act_gamma_t1 = \
            [generator_test.resnet_0_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_0_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_1_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_1_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_2_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_2_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_3_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_3_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_4_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_4_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_5_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_5_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_6_0.AdaFM_0.global_gamma_s[num_task][task].shape[0],
             generator_test.resnet_6_0.AdaFM_1.global_gamma_s[num_task][task].shape[0],
             ]
    elif para == 'beta':
        act_gamma_t1 = \
            [generator_test.resnet_0_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_0_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_1_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_1_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_2_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_2_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_3_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_3_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_4_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_4_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_5_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_5_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_6_0.AdaFM_0.global_beta_s[num_task][task].shape[0],
             generator_test.resnet_6_0.AdaFM_1.global_beta_s[num_task][task].shape[0],
             ]
    return act_gamma_t1


def cumpute_atc_num_v2(generator_test, para='gamma', task=0, num_task=6):
    num_task = num_task - 1
    if para == 'gamma':
        act_gamma_t1 = \
            [generator_test.resnet_0_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_0_0.AdaFM_1.global_num_gamma[task].cpu().data,
             generator_test.resnet_1_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_1_0.AdaFM_1.global_num_gamma[task].cpu().data,
             generator_test.resnet_2_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_2_0.AdaFM_1.global_num_gamma[task].cpu().data,
             generator_test.resnet_3_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_3_0.AdaFM_1.global_num_gamma[task].cpu().data,
             generator_test.resnet_4_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_4_0.AdaFM_1.global_num_gamma[task].cpu().data,
             generator_test.resnet_5_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_5_0.AdaFM_1.global_num_gamma[task].cpu().data,
             generator_test.resnet_6_0.AdaFM_0.global_num_gamma[task].cpu().data,
             generator_test.resnet_6_0.AdaFM_1.global_num_gamma[task].cpu().data,
             ]
    elif para == 'beta':
        act_gamma_t1 = \
            [generator_test.resnet_0_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_0_0.AdaFM_1.global_num_beta[task].cpu().data,
             generator_test.resnet_1_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_1_0.AdaFM_1.global_num_beta[task].cpu().data,
             generator_test.resnet_2_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_2_0.AdaFM_1.global_num_beta[task].cpu().data,
             generator_test.resnet_3_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_3_0.AdaFM_1.global_num_beta[task].cpu().data,
             generator_test.resnet_4_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_4_0.AdaFM_1.global_num_beta[task].cpu().data,
             generator_test.resnet_5_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_5_0.AdaFM_1.global_num_beta[task].cpu().data,
             generator_test.resnet_6_0.AdaFM_0.global_num_beta[task].cpu().data,
             generator_test.resnet_6_0.AdaFM_1.global_num_beta[task].cpu().data,
             ]
    return act_gamma_t1


def get_parameter_num(model, task_id=0):
    p_num = 0

    p_num += model.AdaFM_fc.gamma.shape[1] + model.AdaFM_fc.beta.shape[1] + model.AdaFM_fc.b.shape[0]

    h1 = model.resnet_0_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_0_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_0_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_0_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_0_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 +c
    h1 = model.resnet_0_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_0_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_0_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_0_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_0_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_1_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_1_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_1_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_1_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_1_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_1_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_1_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_1_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_1_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_1_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_2_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_2_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_2_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_2_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_2_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_2_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_2_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_2_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_2_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_2_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_3_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_3_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_3_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_3_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_3_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_3_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_3_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_3_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_3_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_3_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_4_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_4_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_4_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_4_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_4_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_4_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_4_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_4_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_4_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_4_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_5_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_5_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_5_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_5_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_5_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_5_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_5_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_5_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_5_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_5_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c

    h1 = model.resnet_6_0.AdaFM_0.global_gamma_u[0].shape[0]
    w1 = model.resnet_6_0.AdaFM_0.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_6_0.AdaFM_0.global_beta_u[0].shape[0]
    w2 = model.resnet_6_0.AdaFM_0.global_num_beta[task_id].cpu().data
    c = model.resnet_6_0.AdaFM_0.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    h1 = model.resnet_6_0.AdaFM_1.global_gamma_u[0].shape[0]
    w1 = model.resnet_6_0.AdaFM_1.global_num_gamma[task_id].cpu().data
    h2 = model.resnet_6_0.AdaFM_1.global_beta_u[0].shape[0]
    w2 = model.resnet_6_0.AdaFM_1.global_num_beta[task_id].cpu().data
    c = model.resnet_6_0.AdaFM_1.b.shape[0]
    p_num += 2*h1*w1 + w1 + 2*h2*w2 + w2 + c
    return p_num


def load_model_norm(model, is_G=True, is_classCondition=False):
    th_m = torch.tensor(1e-5)
    stdd = 1.0
    dict_all = model.state_dict()
    model_dict = model.state_dict()
    for k, v in dict_all.items():
        if is_G==True:
            if k.find('fc.weight') >= 0:
                w_mu = v.mean([1], keepdim=True)
                w_std = v.std([1], keepdim=True) * stdd
                dict_all[k].data = (v - w_mu)/(w_std)
                dict_all['AdaFM_fc.gamma'].data = w_std.data.t()
                dict_all['AdaFM_fc.beta'].data = w_mu.data.t()
        idt = k.find('conv_0.weight')
        if idt >= 0:
            w_mu = v.mean([2,3], keepdim=True)
            w_std = v.std([2,3], keepdim=True) * stdd
            dict_all[k].data = (v - w_mu)/(w_std)
            dict_all[k[:idt]+'AdaFM_0.style_gama'].data = w_std.data
            dict_all[k[:idt]+'AdaFM_0.style_beta'].data = w_mu.data
        idt = k.find('conv_1.weight')
        if idt >= 0:
            w_mu = v.mean([2, 3], keepdim=True)
            w_std = v.std([2, 3], keepdim=True) * stdd
            dict_all[k].data = (v - w_mu)/(w_std)
            dict_all[k[:idt] + 'AdaFM_1.style_gama'].data = w_std.data
            dict_all[k[:idt] + 'AdaFM_1.style_beta'].data = w_mu.data
        if is_classCondition:
            if k.find('AdaFM_class_bias.weight') >= 0:
                dict_all[k].data = v*0.0

    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


def load_model_norm_svd(model, is_G=True, is_first_task=True):
    # for the first task
    dict_all = model.state_dict()
    model_dict = model.state_dict()
    for k, v in dict_all.items():
        if is_G == True:
            if k.find('fc.weight') >= 0:
                w_mu = v.mean([1], keepdim=True)
                w_std = v.std([1], keepdim=True)
                dict_all[k].data = (v - w_mu) / (w_std)
                dict_all['AdaFM_fc.gamma'].data = w_std.data.t()
                dict_all['AdaFM_fc.beta'].data = w_mu.data.t()
        idt = k.find('conv_0.weight')
        if idt >= 0:
            w_mu = v.mean([2, 3], keepdim=True)
            w_std = v.std([2, 3], keepdim=True)
            dict_all[k].data = (v - w_mu) / (w_std)
            real_rank = howMny_componetes(v.shape[0])
            # gamma
            Ua, Sa, Va = w_std.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_0.gamma_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_0.gamma_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_0.gamma_s2'].data = Sa[jj[:real_rank]].data

            Ua, Sa, Va = w_mu.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_0.beta_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_0.beta_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_0.beta_s2'].data = Sa[jj[:real_rank]].data
        idt = k.find('conv_1.weight')
        if idt >= 0:
            w_mu = v.mean([2, 3], keepdim=True)
            w_std = v.std([2, 3], keepdim=True)
            dict_all[k].data = (v - w_mu) / (w_std)
            real_rank = howMny_componetes(v.shape[0])
            # gamma
            Ua, Sa, Va = w_std.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_1.gamma_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_1.gamma_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_1.gamma_s2'].data = Sa[jj[:real_rank]].data

            Ua, Sa, Va = w_mu.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_1.beta_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_1.beta_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_1.beta_s2'].data = Sa[jj[:real_rank]].data

    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


def load_model_norm_svd_S100(model, is_G=True, is_first_task=True):
    # for the first task
    S_scale_g, S_scale_b = 100.0, 20.0
    dict_all = model.state_dict()
    model_dict = model.state_dict()
    for k, v in dict_all.items():
        if is_G == True:
            if k.find('fc.weight') >= 0:
                w_mu = v.mean([1], keepdim=True)
                w_std = v.std([1], keepdim=True)
                dict_all[k].data = (v - w_mu) / (w_std)
                dict_all['AdaFM_fc.gamma'].data = w_std.data.t()
                dict_all['AdaFM_fc.beta'].data = w_mu.data.t()
        idt = k.find('conv_0.weight')
        if idt >= 0:
            w_mu = v.mean([2, 3], keepdim=True)
            w_std = v.std([2, 3], keepdim=True)
            dict_all[k].data = (v - w_mu) / (w_std)
            real_rank = howMny_componetes(v.shape[0])
            # gamma
            Ua, Sa, Va = w_std.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_0.gamma_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_0.gamma_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_0.gamma_s2'].data = Sa[jj[:real_rank]].data / S_scale_g

            Ua, Sa, Va = w_mu.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_0.beta_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_0.beta_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_0.beta_s2'].data = Sa[jj[:real_rank]].data / S_scale_b
        idt = k.find('conv_1.weight')
        if idt >= 0:
            w_mu = v.mean([2, 3], keepdim=True)
            w_std = v.std([2, 3], keepdim=True)
            dict_all[k].data = (v - w_mu) / (w_std)
            real_rank = howMny_componetes(v.shape[0])
            # gamma
            Ua, Sa, Va = w_std.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_1.gamma_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_1.gamma_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_1.gamma_s2'].data = Sa[jj[:real_rank]].data / S_scale_g

            Ua, Sa, Va = w_mu.data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            dict_all[k[:idt] + 'AdaFM_1.beta_u'].data = Ua[:, jj[:real_rank]].data
            dict_all[k[:idt] + 'AdaFM_1.beta_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                dict_all[k[:idt] + 'AdaFM_1.beta_s2'].data = Sa[jj[:real_rank]].data / S_scale_b

    model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


def load_model_norm_svd_AR(model, dict_all, is_G=True, is_first_task=True):
    # for the first task
    model_dict = model.state_dict()
    dic_choose = {k: v for k, v in dict_all.items() if
                k in model_dict and (k.find('AdaFM_0') == -1 or k.find('AdaFM_1') == -1)}
    for k, v in dict_all.items():
        if k in model_dict:
            model_dict[k].data = v.data
        idt = k.find('conv_0.weight')
        if idt >= 0:
            real_rank = howMny_componetes(v.shape[0])
            # gamma
            Ua, Sa, Va = dict_all[k[:idt] + 'AdaFM_0.style_gama'].data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            model_dict[k[:idt] + 'AdaFM_0.gamma_u'].data = Ua[:, jj[:real_rank]].data
            model_dict[k[:idt] + 'AdaFM_0.gamma_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                model_dict[k[:idt] + 'AdaFM_0.gamma_s2'].data = Sa[jj[:real_rank]].data

            Ua, Sa, Va = dict_all[k[:idt] + 'AdaFM_0.style_beta'].data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            model_dict[k[:idt] + 'AdaFM_0.beta_u'].data = Ua[:, jj[:real_rank]].data
            model_dict[k[:idt] + 'AdaFM_0.beta_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                model_dict[k[:idt] + 'AdaFM_0.beta_s2'].data = Sa[jj[:real_rank]].data
        idt = k.find('conv_1.weight')
        if idt >= 0:
            real_rank = howMny_componetes(v.shape[0])
            # gamma
            Ua, Sa, Va = dict_all[k[:idt] + 'AdaFM_1.style_gama'].data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            model_dict[k[:idt] + 'AdaFM_1.gamma_u'].data = Ua[:, jj[:real_rank]].data
            model_dict[k[:idt] + 'AdaFM_1.gamma_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                model_dict[k[:idt] + 'AdaFM_1.gamma_s2'].data = Sa[jj[:real_rank]].data

            Ua, Sa, Va = dict_all[k[:idt] + 'AdaFM_1.style_beta'].data.squeeze().svd()
            ii, jj = Sa.abs().sort(descending=True)
            model_dict[k[:idt] + 'AdaFM_1.beta_u'].data = Ua[:, jj[:real_rank]].data
            model_dict[k[:idt] + 'AdaFM_1.beta_v'].data = Va[:, jj[:real_rank]].data
            if is_first_task:
                model_dict[k[:idt] + 'AdaFM_1.beta_s2'].data = Sa[jj[:real_rank]].data

    # model_dict.update(dict_all)
    model.load_state_dict(model_dict)
    return model


def chanel_percent(ch, p=[0.95, 0.9, 0.9, 0.8, 0.7]):
    if ch == 64:
        FRAC = p[0] #0.95
    elif ch == 128:
        FRAC = p[1] #0.9
    elif ch==256:
        FRAC = p[2] #0.9
    elif ch == 512:
        FRAC = p[3] #0.8
    elif ch >= 1024:
        FRAC = p[4] #0.7
    return FRAC

def howMny_componetes(ch, is_beta=False, Def=[64, 128, 256, 512, 1024]):
    #Def = [30, 50, 100, 200, 400]
    if is_beta:
        if ch == 64:
            FRAC = Def[0]
        elif ch == 128:
            FRAC = Def[1]
        elif ch == 256:
            FRAC = Def[2]
        elif ch == 512:
            FRAC = Def[3]
        elif ch == 1024:
            FRAC = Def[4]
    else:
        if ch == 64:
            FRAC = Def[0]
        elif ch == 128:
            FRAC = Def[1]
        elif ch == 256:
            FRAC = Def[2]
        elif ch == 512:
            FRAC = Def[3]
        elif ch >= 1024:
            FRAC = Def[4]
    return FRAC


def my_copy(x):
    return x.detach().data * 1.0


