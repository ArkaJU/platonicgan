import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from scripts.trainer.trainer import Trainer
import torch.nn as nn
import torchvision.models as models


class TrainerPlatonic(Trainer):
    def __init__(self, param, dirs, test=False, init=True):
        print("[INFO] setup TrainerPlatonic")

        Trainer.__init__(self, param, dirs, test=test, init=init)
        self.loss_fn = torch.nn.MSELoss()

    def get_graphics_grid_coords_3d(self, z_size, y_size, x_size, coord_dim=-1):
        steps_x = torch.linspace(-1.0, 1.0, x_size)
        steps_y = torch.linspace(1.0, -1.0, y_size)
        steps_z = torch.linspace(1.0, -1.0, z_size)
        z, y, x = torch.meshgrid(steps_z, steps_y, steps_x)
        coords = torch.stack([x, y, z], dim=coord_dim)
        return coords

    def resample(self, volume, indices_rotated):

        if volume.is_cuda:
            indices_rotated = indices_rotated.to('cuda')

        indices_rotated = indices_rotated.permute(0, 2, 3, 4, 1)

        # transform coordinate system
        # flip y and z
        # grid sample expects y- to be up and z- to be front
        indices_rotated[..., 1] = -indices_rotated[..., 1]
        indices_rotated[..., 2] = -indices_rotated[..., 2]
        volume = torch.nn.functional.grid_sample(volume, indices_rotated, mode='bilinear')
        #print(f"Volume_resampled: {volume.shape}")
        return volume

    def rotate(self, volume, rotation_matrix):

        batch_size = volume.shape[0]
        size = volume.shape[2]
        indices = self.get_graphics_grid_coords_3d(size, size, size, coord_dim=0)
        indices = indices.expand(batch_size, 3, size, size, size)
        indices = indices.to(self.param.device)

        indices_rotated = torch.bmm(rotation_matrix, indices.view(batch_size, 3, -1)).view(batch_size, 3, size, size, size)
        return self.resample(volume, indices_rotated)
    
    def get_front_projection(self, rotated_volume):
        raise NotImplementedError

    def _compute_reconstruction_loss(self, fake, real): 
        return self.loss_fn(fake, real) 

    def process_views(self, volume, rotation_matrices, images):

        losses = []
        n_views = rotation_matrices.shape[1]

        for idx in range(n_views):

          rotation_matrix = rotation_matrices[:, idx, :, :] #(B, 3, 3)
          rotated_volume = self.rotate(volume, rotation_matrix)

          real = images[:, idx*4:(idx+1)*4, :, :] #(B, 4, 64, 64)
          fake = self.renderer.render(rotated_volume) #(B, 4, 64, 64)
          loss = self._compute_reconstruction_loss(fake, real)

          self.logger.log_images('{}_{}'.format('view_output', idx), fake)
          losses.append(loss)

        return losses

    def generator_train(self, images, z):
        self.generator.train()
        self.g_optimizer.zero_grad()

        data_loss = torch.tensor(0.0).to(self.param.device)

        fake_volume, rotation_matrices = self.generator(z)  #fake_volume->[B, 4, size, size, size], rotation_matrices->[B, 5, 3, 3]

        losses = self.process_views(fake_volume, rotation_matrices, images)
        g_loss = torch.mean(torch.stack(losses))

        g_loss.backward()
        self.g_optimizer.step()

        ### log
        self.logger.log_scalar('g_2d_loss', g_loss.item())
        #self.logger.log_scalar('g_2d_rec_loss', data_loss.item())
        self.logger.log_volumes('volume', fake_volume)

        #print(f"images.shape: {images.shape}")
        for i in range(images.shape[1]//4):
          #print(f"images[:, i:i+4, :, :].shape: {images[:, i:i+4, :, :].shape}")
          self.logger.log_images(f'image_input_{i}', images[:, i*4:(i+1)*4, :, :])

    def discriminator_train(self, image, volume, z):
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        d_real, _ = self.discriminator(image)
        d_real_loss = self._compute_adversarial_loss(d_real, 1, self.param.training.adversarial_term_lambda_2d)

        with torch.no_grad():
            fake_volume = self.generator(z)

        view, d_fakes, losses, gradient_penalties = self.process_views(fake_volume, image, 0)

        gradient_penalty = torch.mean(torch.stack(gradient_penalties))
        d_fake_loss = torch.mean(torch.stack(losses)) + gradient_penalty
        d_loss = d_real_loss + d_fake_loss

        if self.param.training.loss == 'vanilla':
            d_real = torch.sigmoid(d_real)
            d_fakes = list(map(torch.sigmoid, d_fakes))

            d_real_accuracy = torch.mean(d_real)
            d_fake_accuracy = 1.0 - torch.mean(torch.stack(d_fakes))
            d_accuracy = ((d_real_accuracy + d_fake_accuracy) / 2.0).item()
        else:
            d_accuracy = 0.0

        # only update discriminator if accuracy <= d_thresh
        if d_accuracy <= self.param.training.d_thresh or self.param.training.loss == 'wgangp':
            d_loss.backward()
            self.d_optimizer.step()
            #print("  *Discriminator 2d update*")

        ### log
        #print("  Discriminator2d loss: {:2.4}".format(d_loss))
        self.logger.log_scalar('d_2d_loss', d_loss.item())
        self.logger.log_scalar('d_2d_real', torch.mean(d_real).item())
        self.logger.log_scalar('d_2d_fake', torch.mean(torch.stack(d_fakes)).item())
        self.logger.log_scalar('d_2d_real_loss', d_real_loss.item())
        self.logger.log_scalar('d_2d_fake_loss', d_fake_loss.item())
        self.logger.log_scalar('d_2d_accuracy', d_accuracy)
        self.logger.log_scalar('d_2d_gradient_penalty', gradient_penalty.item())
