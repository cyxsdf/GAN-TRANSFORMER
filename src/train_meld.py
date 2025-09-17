import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # 初始化判别器
    discriminator = None
    disc_optimizer = None

    if hyp_params.modalities != 'LA':
        # 初始化生成器(译者模型)
        if hyp_params.modalities == 'L':
            # 从文本生成音频
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            # 音频判别器
            discriminator = models.Discriminator(hyp_params.orig_d_a,
                                                 hyp_params.a_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_a, hyp_params.a_len)
        elif hyp_params.modalities == 'A':
            # 从音频生成文本
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            # 文本判别器
            discriminator = models.Discriminator(hyp_params.orig_d_l,
                                                 hyp_params.l_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_l, hyp_params.l_len)
        else:
            raise ValueError('Unknown modalities type')

        # 移动生成器到GPU
        if hyp_params.use_cuda:
            translator = translator.cuda()

        # 生成器和判别器优化器
        translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))

        # 损失函数
        trans_criterion = getattr(nn, 'MSELoss')()  # MSE损失
        adv_criterion = nn.BCELoss()  # 对抗损失

    # 初始化主模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 主模型优化器
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 组装训练设置
    if hyp_params.modalities != 'LA':
        settings = {
            'model': model,
            'translator': translator,
            'translator_optimizer': translator_optimizer,
            'discriminator': discriminator,
            'disc_optimizer': disc_optimizer,
            'trans_criterion': trans_criterion,
            'adv_criterion': adv_criterion,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }
    else:
        settings = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    # 提取生成器、判别器和损失函数
    translator = None
    discriminator = None
    disc_optimizer = None
    trans_criterion = None
    adv_criterion = None

    if hyp_params.modalities != 'LA':
        trans_criterion = settings['trans_criterion']
        adv_criterion = settings['adv_criterion']
        discriminator = settings['discriminator']
        disc_optimizer = settings['disc_optimizer']
        translator = settings['translator']
        translator_optimizer = settings['translator_optimizer']
    else:
        translator = None

    def train(model, translator, optimizer, criterion):
        epoch_loss = 0
        model.train()

        if hyp_params.modalities != 'LA':
            translator.train()
            discriminator.train()  # 判别器设为训练模式

        num_batches = hyp_params.n_train // hyp_params.batch_size
        start_time = time.time()

        for i_batch, (audio, text, masks, labels) in enumerate(train_loader):
            # 梯度清零
            model.zero_grad()
            if hyp_params.modalities != 'LA':
                translator.zero_grad()
                disc_optimizer.zero_grad()  # 判别器梯度清零

            # 数据移至GPU
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

            # 应用掩码
            masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
            if hyp_params.dataset == 'meld_senti':
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)  # meld_sentiment
            else:
                masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)  # meld_emotion
            text = text * masks_text
            audio = audio * masks_audio

            batch_size = text.size(0)
            net = nn.DataParallel(model) if hyp_params.distribute else model
            trans_loss = 0  # 生成损失初始化
            fake_a, fake_l = None, None

            # 生成缺失模态并计算GAN损失
            if hyp_params.modalities != 'LA':
                trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                # 生成缺失模态
                if hyp_params.modalities == 'L':
                    fake_a = trans_net(text, audio, 'train')
                    # MSE损失
                    mse_loss = trans_criterion(fake_a, audio)

                elif hyp_params.modalities == 'A':
                    fake_l = trans_net(audio, text, 'train')
                    # MSE损失
                    mse_loss = trans_criterion(fake_l, text)

                # 训练判别器
                real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(batch_size, 1)

                # 真实样本损失
                if hyp_params.modalities == 'L':
                    real_pred = discriminator(audio)
                else:  # A
                    real_pred = discriminator(text)
                real_loss = adv_criterion(real_pred, real_labels)

                # 生成样本损失 (分离生成器，不更新生成器)
                if hyp_params.modalities == 'L':
                    fake_pred = discriminator(fake_a.detach())
                else:  # A
                    fake_pred = discriminator(fake_l.detach())
                fake_loss = adv_criterion(fake_pred, fake_labels)

                # 判别器总损失
                disc_loss = (real_loss + fake_loss) * 0.5
                disc_loss.backward()
                disc_optimizer.step()

                # 训练生成器的对抗损失
                if hyp_params.modalities == 'L':
                    gen_pred = discriminator(fake_a)
                else:  # A
                    gen_pred = discriminator(fake_l)
                adv_loss = adv_criterion(gen_pred, real_labels)

                # 组合损失：MSE + 对抗损失
                trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight

            # 模型前向传播
            if hyp_params.modalities != 'LA':
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio)
                else:
                    raise ValueError('Unknown modalities type')
            else:
                preds, _ = net(text, audio)

            # 计算主模型损失
            raw_loss = criterion(preds.transpose(1, 2), labels)

            # 组合损失
            if hyp_params.modalities != 'LA':
                combined_loss = raw_loss + trans_loss
            else:
                combined_loss = raw_loss

            # 反向传播和参数更新
            combined_loss.backward()

            # 梯度裁剪
            if hyp_params.modalities != 'LA':
                torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                translator_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            # 累计损失
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        model.eval()
        if hyp_params.modalities != 'LA':
            translator.eval()
            discriminator.eval()  # 判别器设为评估模式

        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (audio, text, masks, labels) in enumerate(loader):
                # 数据移至GPU
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, masks, labels = text.cuda(), audio.cuda(), masks.cuda(), labels.cuda()

                # 应用掩码
                masks_text = masks.unsqueeze(-1).expand(-1, 33, 600)
                if hyp_params.dataset == 'meld_senti':
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 600)  # meld_sentiment
                else:
                    masks_audio = masks.unsqueeze(-1).expand(-1, 33, 300)  # meld_emotion
                text = text * masks_text
                audio = audio * masks_audio

                batch_size = text.size(0)
                net = nn.DataParallel(model) if hyp_params.distribute else model
                fake_a, fake_l = None, None

                # 生成缺失模态
                if hyp_params.modalities != 'LA':
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                    if not test:  # 验证模式
                        if hyp_params.modalities == 'L':
                            fake_a = trans_net(text, audio, 'valid')
                            trans_loss = trans_criterion(fake_a, audio)
                        elif hyp_params.modalities == 'A':
                            fake_l = trans_net(audio, text, 'valid')
                            trans_loss = trans_criterion(fake_l, text)
                    else:  # 测试模式，自回归生成
                        if hyp_params.modalities == 'L':
                            # 生成音频模态
                            fake_a = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.a_len):
                                if i == 0:
                                    fake_a_token = trans_net(text, audio, 'test', eval_start=True)[:, [-1]]
                                else:
                                    fake_a_token = trans_net(text, fake_a, 'test')[:, [-1]]
                                fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                        elif hyp_params.modalities == 'A':
                            # 生成文本模态
                            fake_l = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.l_len):
                                if i == 0:
                                    fake_l_token = trans_net(audio, text, 'test', eval_start=True)[:, [-1]]
                                else:
                                    fake_l_token = trans_net(audio, fake_l, 'test')[:, [-1]]
                                fake_l = torch.cat((fake_l, fake_l_token), dim=1)

                # 模型预测
                if hyp_params.modalities != 'LA':
                    if hyp_params.modalities == 'L':
                        preds, _ = net(text, fake_a)
                    elif hyp_params.modalities == 'A':
                        preds, _ = net(fake_l, audio)
                    else:
                        raise ValueError('Unknown modalities type')
                else:
                    preds, _ = net(text, audio)

                # 计算损失
                raw_loss = criterion(preds.transpose(1, 2), labels)
                if hyp_params.modalities != 'LA' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss

                total_loss += combined_loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    # 打印模型参数信息
    if hyp_params.modalities != 'LA':
        mgm_parameter = sum([param.nelement() for param in translator.parameters()])
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')

        # 判别器参数
        disc_parameter = sum([param.nelement() for param in discriminator.parameters()])
        print(f'Trainable Parameters for Discriminator: {disc_parameter}...')

    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')

    # 训练循环
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()

        # 训练模型
        train(model, translator, optimizer, criterion)

        # 验证模型
        val_loss, _, _, _ = evaluate(model, translator, criterion, test=False)
        end = time.time()
        duration = end - start

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            if hyp_params.modalities != 'LA':
                save_model(hyp_params, translator, name='TRANSLATOR')
                save_model(hyp_params, discriminator, name='DISCRIMINATOR')

            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型进行测试
    if hyp_params.modalities != 'LA':
        translator = load_model(hyp_params, name='TRANSLATOR')
        discriminator = load_model(hyp_params, name='DISCRIMINATOR')

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, mask = evaluate(model, translator, criterion, test=True)

    # 评估结果
    acc = eval_meld(results, truths, mask)

    return acc
