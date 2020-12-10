import argparse
import os, sys
from os import path
import time
import copy
import torch
from torch import nn
import numpy as np
import random


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(999)

# from torchsummary import summary
import shutil
import scipy.io as sio
from gan_training import utils
from gan_training.utils_model_load import *
from gan_training.train_OWM import Trainer, update_average

from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config_OWM import (
    load_config, build_models, build_optimizers, build_lr_scheduler, build_models_PRE,
)

''' ===================--- Set the traning mode ---==========================
DATA: going to train
DATA_FIX: used as a fixed pre-trained model
============================================================================='''
seed_torch(999)
DATA_FIX = 'CELEBA'
Num_epoch = 60000//(7200//16) + 1 

CONS_lowrank = 0.01
# task_id = 5
for task_id in range(6):

    # main_path = 'F:/contral_kernel_results_remote/OWM_GAN/code/'
    # load_dir = 'F:/RUN_CODE_OUT/GAN_stability/pretrained/weights/saved_data/'
    # out_path = 'F:/RUN_CODE_OUT/OWM' +'/Six_butterfly_task%d_lowrank/'%task_id
    
    
    main_path = '/hpc/home/mz149/code/OWM_GAN/'
    load_dir = '/hpc/home/mz149/saved_model/'
    out_path = "/hpc/group/carin/mz149/code/" + '/Six_butterfly_task%d_lowrank_highLayers/'%task_id
    
    config_path = main_path+'/configs/' +'Flowers'+ '_celeba.yaml'
    
    
    config = load_config(config_path, 'configs/default.yaml')
    config['generator']['name'] = 'resnet4_AdaFM_accumulate_lowrank_highLayers'
    config['discriminator']['name'] = 'resnet4_AdaFM_accumulate_lowrank_highLayers'
    config['training']['out_dir'] = out_path
    if not os.path.isdir(config['training']['out_dir']):
        os.makedirs(config['training']['out_dir'])
    
    if task_id == 0:
        DATA = 'but0'
        NNN = 1300
        # image_path = 'F:/download_data/Image_DATA/butterfly/n02276258/'
        image_path = "/hpc/group/carin/mz149/data/butterfly/n02276258/"
        image_test = image_path
    elif task_id == 1:
        DATA = 'but1'
        NNN = 1300
        # image_path = 'F:/download_data/Image_DATA/butterfly/n02279972/'
        image_path = "/hpc/group/carin/mz149/data/butterfly/n02279972/"
        image_test = image_path
    elif task_id == 2:
        DATA = 'but2'
        NNN = 1300
        # image_path = 'F:/download_data/Image_DATA/butterfly/n02277742/'
        image_path = "/hpc/group/carin/mz149/data/butterfly/n02277742/"
        image_test = image_path
    elif task_id == 3:
        DATA = 'but3'
        NNN = 1300
        # image_path = 'F:/download_data/Image_DATA/butterfly/n02280649/'
        image_path = "/hpc/group/carin/mz149/data/butterfly/n02280649/"
        image_test = image_path
    elif task_id == 4:
        DATA = 'but4'
        # image_path = 'F:/download_data/Image_DATA/butterfly/n02281406/'
        image_path = "/hpc/group/carin/mz149/data/butterfly/n02281406/"
        image_test = image_path
        NNN = 1300
    elif task_id == 5:
        DATA = 'but5'
        # image_path = 'F:/download_data/Image_DATA/butterfly/n02281787/'
        image_path = "/hpc/group/carin/mz149/data/butterfly/n02281787/"
        image_test = image_path
        NNN = 1300
    
    config['data']['train_dir'] = image_path
    config['data']['test_dir'] = image_test
    
    
    if 1:
        # Short hands
        batch_size = config['training']['batch_size']
        d_steps = config['training']['d_steps']
        restart_every = config['training']['restart_every']
        inception_every = config['training']['inception_every']
        save_every = config['training']['save_every']
        backup_every = config['training']['backup_every']
        sample_nlabels = config['training']['sample_nlabels']
        dim_z = config['z_dist']['dim']
    
        out_dir = config['training']['out_dir']
        checkpoint_dir = path.join(out_dir, 'chkpts')
    
        # Create missing directories
        if not path.exists(out_dir):
            os.makedirs(out_dir)
        if not path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')
    
        # Logger
        checkpoint_io = CheckpointIO(
            checkpoint_dir=checkpoint_dir
        )
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Dataset
        train_dataset, nlabels = get_dataset(
            name=config['data']['type'],
            data_dir=config['data']['train_dir'],
            size=config['data']['img_size'],
            lsun_categories=config['data']['lsun_categories_train']
        )
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=config['training']['nworkers'],
                shuffle=True, pin_memory=True, sampler=None, drop_last=True
        )
        test_dataset, _ = get_dataset(
            name=config['data']['type'],
            data_dir=config['data']['test_dir'],
            size=128,
            lsun_categories=config['data']['lsun_categories_train']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
        )
        print('train_dataset=', train_dataset)
    
        nlabels = min(nlabels, config['data']['nlabels'])
        sample_nlabels = min(nlabels, sample_nlabels)
    
        # Distributions
        ydist = get_ydist(nlabels, device=device)
        zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                          device=device)
    
        # Save for tests
        ntest = 10
        x_real, ytest = utils.get_nsamples(train_loader, ntest)
        ytest.clamp_(None, nlabels - 1)
        ytest = ytest.to(device)
        ztest = zdist.sample((ntest,)).to(device)
        utils.save_images(x_real, path.join(out_dir, 'real.png'))
    
        # Create models
        ''' --------- Choose the fixed layer ---------------'''
        generator, discriminator = build_models(config)
    
        dict_G = torch.load(load_dir + DATA_FIX + 'Pre_generator')
        generator = model_equal_all(generator, dict_G)
        generator = load_model_norm(generator)
        dict_D = torch.load(load_dir + DATA_FIX + 'Pre_discriminator')
        discriminator = model_equal_all(discriminator, dict_D)
        generator, discriminator = generator.to(device), discriminator.to(device)
        # Logger
        logger = Logger(
            log_dir=path.join(out_dir, 'logs'),
            img_dir=path.join(out_dir, 'imgs'),
            monitoring=config['training']['monitoring'],
            monitoring_dir=path.join(out_dir, 'monitoring')
        )
        # with torch.no_grad():
        #     x,_ = generator(ztest, ytest, task_id=-1)
        #     logger.add_imgs(x, 'all', 5, nrow=2)
        # discriminator = load_model_norm(discriminator, is_G=False)
        if task_id > 0:
            if task_id >= 1:
                model_file = "/hpc/group/carin/mz149/code/Six_butterfly_task0_lowrank/models/but0_00024999_Pre_generator"
                dict_G = torch.load(model_file)
                generator = model_equal_all(generator, dict_G)
                generator(ztest, ytest, UPDATE_GLOBAL=True,device=device)
            if task_id >= 2:
                model_file = "/hpc/group/carin/mz149/code/Six_butterfly_task1_lowrank_highLayers/models/but1_00024999_Pre_generator"
                dict_G = torch.load(model_file)
                generator = model_equal_all(generator, dict_G)
                generator(ztest, ytest, UPDATE_GLOBAL=True,device=device)
            if task_id >= 3:
                model_file = "/hpc/group/carin/mz149/code/Six_butterfly_task2_lowrank_highLayers/models/but2_00024999_Pre_generator"
                dict_G = torch.load(model_file)
                generator = model_equal_all(generator, dict_G)
                generator(ztest, ytest, UPDATE_GLOBAL=True,device=device)
            if task_id >= 4:
                model_file = "/hpc/group/carin/mz149/code/Six_butterfly_task3_lowrank_highLayers/models/but3_00024999_Pre_generator"
                dict_G = torch.load(model_file)
                generator = model_equal_all(generator, dict_G)
                generator(ztest, ytest, UPDATE_GLOBAL=True,device=device)
            if task_id >= 5:
                model_file = "/hpc/group/carin/mz149/code/Six_butterfly_task4_lowrank_highLayers/models/but4_00024999_Pre_generator"
                dict_G = torch.load(model_file)
                generator = model_equal_all(generator, dict_G)
                generator(ztest, ytest, UPDATE_GLOBAL=True,device=device)
            # generator = load_model_norm_svd(generator, is_first_task=False)
    
        # with torch.no_grad():
        #     x,_ = generator(ztest, ytest, task_id=0)
        #     logger.add_imgs(x, 'all', 7, nrow=2)
        # print('generator.resnet_0_0.AdaFM_0.global_b ======== ', generator.resnet_0_0.AdaFM_0.global_b)
    
        for name, param in generator.named_parameters():
            if name.find('AdaFM_') >= 0:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
        for name, param in discriminator.named_parameters():
            if name.find('AdaFM_') >= 0:
                param.requires_grad = True
            elif name.find('fc') >= 0:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
        # Put models on gpu if needed
        g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)
    
        # summary(generator, input_size=[(256,), (1,)])
        # summary(discriminator, input_size=[(3, 128, 128), (1,)])
    
        # Register modules to checkpoint
        checkpoint_io.register_modules(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
        )
    
    
    
        # Test generator
        if config['training']['take_model_average']:
            generator_test = copy.deepcopy(generator)
            checkpoint_io.register_modules(generator_test=generator_test)
        else:
            generator_test = generator
    
        # Evaluator
        # evaluator = Evaluator(generator_test, zdist, ydist,
        #                       batch_size=batch_size, device=device)
        x_real_FID, _ = utils.get_nsamples(test_loader, NNN)
        evaluator = Evaluator(generator_test, zdist, ydist,
                              batch_size=batch_size, device=device,
                              fid_real_samples=x_real_FID,
                              inception_nsamples=NNN, fid_sample_size=NNN)
    
    
        it = -1
        epoch_idx = -1
    
        # Reinitialize model average if needed
        if (config['training']['take_model_average']
                and config['training']['model_average_reinit']):
            update_average(generator_test, generator, 0.)
    
        # Learning rate anneling
        g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
        d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)
    
        # Trainer
        trainer = Trainer(
            generator, discriminator, g_optimizer, d_optimizer,
            gan_type=config['training']['gan_type'],
            reg_type=config['training']['reg_type'],
            reg_param=config['training']['reg_param'],
            D_fix_layer=config['discriminator']['layers'],
            CONS_lowrank=CONS_lowrank,
        )
    
        # Training loop
        print('Start training...')
        save_dir = config['training']['out_dir'] + '/models/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        get_parameter_number(generator)
        get_parameter_number(discriminator)
    
    inception_mean_all = []
    inception_std_all = []
    fid_all = []
    
    tstart = time.time()
    
    for epoch_idx in range(Num_epoch):
        # epoch_idx += 1
        print('Start epoch %d...' % epoch_idx)
    
        for x_real, y in train_loader:
            it += 1
            g_scheduler.step()
            d_scheduler.step()
    
            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']
    
            x_real, y = x_real.to(device), y.to(device)
            y.clamp_(None, nlabels-1)
    
            # Generators updates
            z = zdist.sample((batch_size,)).to(device)
            gloss, x_fake, l1_loss = trainer.generator_trainstep(y, z, Iterr=it, is_lowrank=True)
    
            if it>10 and config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])
    
            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, x_fake, Iterr=it)
    
            with torch.no_grad():
    
                # (i) Sample if necessary
                if (it % config['training']['sample_every']) == 0:
                    d_fix, d_update = discriminator.conv_img.weight[1, 1, 1, 1], discriminator.fc.weight[0, 1]
                    g_fix, g_update = generator.conv_img.weight[1, 1, 1, 1], 0.0
    
                    print('[epoch %0d, it %4d] g_loss = %.4f/%.4f, d_loss = %.4f, reg=%.4f, d_fix=%.4f, d_update=%.4f, g_fix=%.4f, g_update=%.4f, time=%.2f'
                          % (epoch_idx, it, gloss, l1_loss, dloss, reg, d_fix, d_update, g_fix, g_update, time.time()-tstart))
                    tstart = time.time()
                    # print('Creating samples...')
                    x, _ = generator_test(ztest, ytest)
                    logger.add_imgs(x, 'all', it, nrow=2)
    
                # (ii) Compute inception if necessary
                if ((it + 1) % 5000) == 0:
                    exit_real_ms = False
                    inception_mean, inception_std, fid = evaluator.compute_inception_score(is_FID=True,
                                                                                           exit_real_ms=exit_real_ms)
    
                    inception_mean_all.append(inception_mean)
                    inception_std_all.append(inception_std)
                    fid_all.append(fid)
                    print('test it %d: IS: mean %.2f, std %.2f, FID: mean %.2f, time: %2f' % (
                        it, inception_mean, inception_std, fid, time.time() - tstart))
    
                    FID = np.stack(fid_all)
                    Inception_mean = np.stack(inception_mean_all)
                    Inception_std = np.stack(inception_std_all)
                    sio.savemat(out_path + DATA + 'base_FID_IS.mat', {'FID': FID,
                                                           'Inception_mean': Inception_mean,
                                                           'Inception_std': Inception_std})
    
                # (iii) Backup if necessary
                if ((it + 1) % backup_every) == 0:
                    print('Saving backup...')
                    TrainModeSave = DATA + '_%08d_' % it
                    generator_test_part = save_adafm_only(generator_test)
                    torch.save(generator_test_part, save_dir + TrainModeSave + 'Pre_generator')
                if it+1 == 60000:
                    TrainModeSave = DATA + '_%08d_' % it
                    discriminator_part = save_adafm_only(discriminator, is_G=False)
                    torch.save(discriminator_part, save_dir + TrainModeSave + 'Pre_discriminator')
    
