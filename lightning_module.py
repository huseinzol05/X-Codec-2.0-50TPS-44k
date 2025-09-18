import os
 
import random
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss, MultiResolutionSTFTLoss
from common.schedulers import WarmupLR
from transformers import AutoModel
from modeling_xcodec2 import XCodec2Model
import sys



class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ocwd = hydra.utils.get_original_cwd()
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def construct_model(self):

        self.model = XCodec2Model.from_pretrained("HKUSTAudio/xcodec2", ignore_mismatched_sizes=True).train()
        for name, p in self.model.named_parameters():
            if 'generator' not in name:
                p.requires_grad = False
        
        # 初始化 MultiPeriod Discriminator
        mpdcfg = self.cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
            periods=mpdcfg.periods,
            max_downsample_channels=mpdcfg.max_downsample_channels,
            channels=mpdcfg.channels,
            channel_increasing_factor=mpdcfg.channel_increasing_factor,
        )

        # 初始化 Spectral Discriminator
        mstftcfg = self.cfg.model.mstft
        self.spec_discriminator = SpecDiscriminator(
            stft_params=mstftcfg.stft_params,
            in_channels=mstftcfg.in_channels,
            out_channels=mstftcfg.out_channels,
            kernel_sizes=mstftcfg.kernel_sizes,
            channels=mstftcfg.channels,
            max_downsample_channels=mstftcfg.max_downsample_channels,
            downsample_scales=mstftcfg.downsample_scales,
            use_weight_norm=mstftcfg.use_weight_norm,
        )

    def construct_criteria(self):
        cfg = self.cfg.train
        self.criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            self.criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.preprocess.audio.sr)
        if cfg.use_stft_loss:
            self.criteria['stft_loss'] = MultiResolutionSTFTLoss(
                fft_sizes=cfg.stft_loss_params.fft_sizes,
                hop_sizes=cfg.stft_loss_params.hop_sizes,
                win_sizes=cfg.stft_loss_params.win_lengths
            )
        if cfg.use_feat_match_loss:
            self.criteria['fm_loss'] = nn.L1Loss()
        self.criteria['gan_loss'] = GANLoss()
        self.criteria['l1_loss'] = nn.L1Loss()
        self.criteria['l2_loss'] = nn.MSELoss()
        print(self.criteria)

    def forward(self, batch):
        wav = batch['wav']
        wav_24k = batch['wav_24k']
        feats = batch['feats']

        y_ = self.model(input_waveform=wav, input_features=feats[:,0])
        y = wav_24k.unsqueeze(1)

        print('y', y_.shape, y.shape)

        y_ = y_[:,:,:y.shape[2]]

        output = {
            'gt_wav': y,
            'gen_wav': y_,
 
        }
        return output

    @torch.inference_mode()
    def inference(self, wav):
        y_ = self.model(wav)
        return y_

    def compute_disc_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        p = self.discriminator(y)
        p_ = self.discriminator(y_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if hasattr(self, 'spec_discriminator'):
            sd_p = self.spec_discriminator(y)
            sd_p_ = self.spec_discriminator(y_)

            for i in range(len(sd_p)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(sd_p[i][-1], sd_p_[i][-1])
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss

        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }
        return output

    def compute_gen_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        # perceptual_se_loss_l2 = output['perceptual_se_loss_l2']
        # x_feat_recon_loss = output['x_feat_recon_loss']
        gen_loss = 0.0
        self.set_discriminator_gradients(False)
        output_dict = {}
        cfg = self.cfg.train

        # Mel spectrogram loss
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output_dict['mel_loss'] = mel_loss

        # GAN loss
        p_ = self.discriminator(y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
        if hasattr(self, 'spec_discriminator'):
            sd_p_ = self.spec_discriminator(y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(sd_p_[i][-1]))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        output_dict['adv_loss'] = adv_loss

        # Feature Matching loss
        if cfg.use_feat_match_loss:
            fm_loss = 0.0
            with torch.no_grad():
                p = self.discriminator(y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            output_dict['fm_loss'] = fm_loss
            if hasattr(self, 'spec_discriminator'):
                spec_fm_loss = 0.0
                with torch.no_grad():
                    sd_p = self.spec_discriminator(y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output_dict['spec_fm_loss'] = spec_fm_loss

        # Perceptual loss
        # output_dict['perceptual_se_loss_l2'] = perceptual_se_loss_l2
        # gen_loss += output_dict['perceptual_se_loss_l2'] * cfg.lambdas.lambda_perceptual_loss
        
        self.set_discriminator_gradients(True)
        output_dict['gen_loss'] = gen_loss
        return output_dict

    def training_step(self, batch, batch_idx):
        output = self(batch)

        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()

        # 训练判别器
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss']
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        self.clip_gradients(
            disc_opt,
            gradient_clip_val=self.cfg.train.disc_grad_clip,
            gradient_clip_algorithm='norm'
        )
        disc_opt.step()
        disc_sche.step()

        # 训练生成器
        gen_losses = self.compute_gen_loss(batch, output)
        gen_loss = gen_losses['gen_loss']
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(
            gen_opt,
            gradient_clip_val=self.cfg.train.gen_grad_clip,
            gradient_clip_algorithm='norm'
        )
        gen_opt.step()
        gen_sche.step()

        # 记录损失
        self.log_dict(
            disc_losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.cfg.dataset.train.batch_size,
            sync_dist=True
        )
        self.log_dict(
            gen_losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.cfg.dataset.train.batch_size,
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        pass

 

    def test_step(self, batch, batch_idx):
        # 您可以在此处实现测试逻辑
        pass

    def configure_optimizers(self):
        from itertools import chain

        # 判别器参数
        disc_params = self.discriminator.parameters()
        # if hasattr(self, 'spec_discriminator'):
        disc_params = chain(disc_params, self.spec_discriminator.parameters())

        gen_params = (p for p in self.model.parameters() if p.requires_grad)

        # 优化器
        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        # 学习率调度器
        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)

        print(f'Generator optim: {gen_opt}')
        print(f'Discriminator optim: {disc_opt}')

        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.discriminator.parameters():
            p.requires_grad = flag

        if hasattr(self, 'spec_discriminator'):
            for p in self.spec_discriminator.parameters():
                p.requires_grad = flag
