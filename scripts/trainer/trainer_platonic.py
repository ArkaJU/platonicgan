import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from scripts.trainer.trainer import Trainer
from scripts.utils.io import volume_to_raw
import torch.nn as nn
import torchvision.models as models


class TrainerPlatonic(Trainer):
    def __init__(self, param, dirs, test=False, init=True):
        print("[INFO] setup TrainerPlatonic")

        Trainer.__init__(self, param, dirs, test=test, init=init)
         
        self.discriminator = self.discriminator_2d
        self.d_optimizer = self.d_optimizer_2d

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
        return torch.nn.MSELoss()(fake, real) 

    def process_views(self, volume, rotation_matrices, images, target):

        losses_rec = []
        losses_adv = []
        d_fakes = []
        n_views = rotation_matrices.shape[1]

        #rotation using output matrices
        for idx in range(n_views):
          rotation_matrix = rotation_matrices[:, idx, :, :] #(B, 3, 3)
          rotated_volume = self.rotate(volume, rotation_matrix)


          if self.param.renderer.type == 'visual_hull' or self.param.renderer.type == 'absorption_only':
            real = images[:, idx, :, :]            
            real = real.unsqueeze(1)                #(B, 1, 64, 64)
          elif self.param.renderer.type == 'emission_absorption':
            real = images[:, idx*4:(idx+1)*4, :, :] #(B, 4, 64, 64)
          
          fake = self.renderer.render(rotated_volume) #(B, 4, 64, 64) or (B, 1, 64, 64)
          
          loss_rec = self._compute_reconstruction_loss(fake, real)  
          self.logger.log_images('{}_{}'.format(idx, 'view_output'), fake)
          losses_rec.append(loss_rec)
          
        #random rotation 
        for idx in range(n_views):
          fake = self.renderer.render(self.transform.rotate_random(volume))
          d_fake = self.discriminator(fake)
          loss_adv = self._compute_adversarial_loss(d_fake, target, self.param.training.adversarial_term_lambda_2d)
          self.d_optimizer.zero_grad()
          d_fakes.append(d_fake)
          self.logger.log_images('{}_{}'.format('random_fake_rotation', idx), fake)
          losses_adv.append(loss_adv)
        
        # x = np.array([[0.866, -0.5,    0],
        #               [0.5,    0.866,  0 ], 
        #               [0,      0,      1]])
        x = np.array([[0,  -1,  0],
                      [1,   0,  0], 
                      [0,   0,  1]])
        x = x[np.newaxis, :, :]

        x = torch.Tensor(np.repeat(x, self.param.training.batch_size, 0)).to('cuda') #(B, 3, 3)
        rotation_matrix = torch.bmm(rotation_matrices[:, 2, :, :], x)
        rotated_volume = self.rotate(volume, rotation_matrix)
        fake = self.renderer.render(rotated_volume) #(B, 4, 64, 64)
        
        self.logger.log_images('{}'.format('test_output'), fake)
        return d_fakes, losses_rec, losses_adv
        #return losses_rec

    def generator_train(self, images, z):
        self.generator.train()
        self.g_optimizer.zero_grad()

        data_loss = torch.tensor(0.0).to(self.param.device)

        fake_volume, rotation_matrices = self.generator(z)  #fake_volume->[B, 4, size, size, size], rotation_matrices->[B, 5, 3, 3]
        torch.save(rotation_matrices, '/content/rotation_matrices.pt')
        _, losses_rec, losses_adv = self.process_views(fake_volume, rotation_matrices, images, target=1)
        #losses_rec = self.process_views(fake_volume, rotation_matrices, images, target=1)
        losses_rec = torch.mean(torch.stack(losses_rec))
        losses_adv = torch.mean(torch.stack(losses_adv))
        
        alpha= 0.5
        g_loss = losses_rec + alpha*losses_adv

        g_loss.backward()
        self.g_optimizer.step()

        ### log
        self.logger.log_graph(self.generator, z)
        self.logger.log_scalar('g_loss', g_loss.item())
        self.logger.log_scalar('g_rec_loss', losses_rec.item())
        self.logger.log_scalar('g_adv_loss', losses_adv.item())
        self.logger.log_volumes('volume', fake_volume)
        #print(fake_volume[0].shape)
        #volume_to_raw(fake_volume[0].permute(3,2,1,0).detach().cpu().clone().numpy(), path='/content', name='fake_volume')
        np.save('/content/volume.npy', fake_volume[0].permute(3,2,1,0).detach().cpu().clone().numpy())

        if self.param.renderer.type == 'visual_hull' or self.param.renderer.type == 'absorption_only':
          for i in range(images.shape[1]):
            self.logger.log_images(f'{i}_image_input', images[:, i, :, :].unsqueeze(1))
        
        elif self.param.renderer.type == 'emission_absorption':
          for i in range(images.shape[1]//4):
            self.logger.log_images(f'{i}_image_input', images[:, i*4:(i+1)*4, :, :])

    def discriminator_train(self, images, z):
        self.discriminator.train()
        self.d_optimizer.zero_grad()
        
        d_real = []

        if self.param.renderer.type == 'visual_hull' or self.param.renderer.type == 'absorption_only':
          for i in range(images.shape[1]):
            real = images[:, i, :, :]
            real = real.unsqueeze(1)
            d = self.discriminator(real)
            d_real.append(d)

        elif self.param.renderer.type == 'emission_absorption':
          for i in range(images.shape[1]//4):
            real = images[:, i*4:(i+1)*4, :, :]
            d = self.discriminator(real)
            d_real.append(d)

        d_real = torch.cat(d_real, axis=0)   #(B*5, 1, 1, 1)
        torch.save(d_real, '/content/d_real.pt')
        target = 1
        d_real_loss = self._compute_adversarial_loss(d_real, target, self.param.training.adversarial_term_lambda_2d)

        with torch.no_grad():
            fake_volume, rotation_matrices = self.generator(z)

        d_fakes, _, losses_adv = self.process_views(fake_volume, rotation_matrices, images, target=0)
        d_fake_loss = torch.mean(torch.stack(losses_adv)) 
        d_loss = d_real_loss + d_fake_loss

        d_real = torch.sigmoid(d_real)

        d_fakes = torch.cat(d_fakes, axis=0)  #(B*5, 1, 1, 1)
        torch.save(d_fakes, '/content/d_fakes.pt')
        d_fakes = torch.sigmoid(d_fakes)

        d_real_accuracy = torch.mean(d_real)
        d_fake_accuracy = 1.0 - torch.mean(d_fakes)
        d_accuracy = ((d_real_accuracy + d_fake_accuracy) / 2.0).item()


        # only update discriminator if accuracy <= d_thresh
        d_thresh = 0.8
        if d_accuracy <= self.param.training.d_thresh:
            d_loss.backward()
            self.d_optimizer.step()
            #print("  *Discriminator 2d update*")

        ### log
        self.logger.log_graph(self.discriminator, real)
        self.logger.log_scalar('d_loss', d_loss.item())
        self.logger.log_scalar('d_loss_real', d_real_loss.item())
        self.logger.log_scalar('d_loss_fake', d_fake_loss.item())

        self.logger.log_scalar('d_accuracy', d_accuracy)
        self.logger.log_scalar('d_accuracy_real', d_real_accuracy.item())
        self.logger.log_scalar('d_accuracy_fake', d_fake_accuracy.item())
        #self.logger.log_scalar('d_2d_gradient_penalty', gradient_penalty.item())
