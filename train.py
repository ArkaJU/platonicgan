import os
import matplotlib
import time, torch
from scripts.trainer import TrainerPlatonic, Trainer3D, TrainerPlatonic3D
from scripts.tests.tests import run_tests
from scripts.utils.config import load_config, make_dirs
import scripts.utils.utils as utils

# training setup
param = load_config()
dirs = make_dirs(param)

if param.mode == 'platonic':
    trainer = TrainerPlatonic(param, dirs)
elif param.mode == 'platonic_3D':
    trainer = TrainerPlatonic3D(param, dirs)
elif param.mode == '3D':
    trainer = Trainer3D(param, dirs)

# if param.tests.activate:
#     run_tests(trainer)

print('[Training] training start:')

for epoch in range(1, param.training.n_epochs + 1):
    start_time = time.time()
    if trainer.logger.iteration % param.logger.log_files_every==0:
      print("[iter {} name: {} mode: {} IF: {}]".format(trainer.logger.iteration, param.name, param.mode, param.renderer.type))
    
    for idx, (images, object_id) in enumerate(trainer.datal_loader_train):

        images = images.to(param.device)
        #print(images.shape)                 #(B, 20, 64, 64) or (B, 5, 64, 64)
        z = trainer.encoder_train(images)    #(B, 200)
        #print(z.shape) 
        trainer.generator_train(images, z)
        trainer.discriminator_train(images, z)

        trainer.logger.log_checkpoint(trainer.models, trainer.optimizers)
        for model in trainer.models:
          trainer.logger.log_gradients(model)

    trainer.logger.step()
print('Training finished!...')
