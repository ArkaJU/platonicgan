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

    for idx, (images, object_id) in enumerate(trainer.datal_loader_train):
        if trainer.logger.iteration%50==0:
          print("[iter {} name: {} mode: {} IF: {}]".format(trainer.logger.iteration, param.name, param.mode, param.renderer.type))
        print(f'train: {idx}')
        images = images.to(param.device)
        volume = volume.to(param.device)
        
        z = trainer.encoder_train(images)
        print("#"*100)
        trainer.generator_train(images, z)
        print("$"*100)
        # trainer.discriminator_train(image, volume, z)
        # print("%"*100)
        trainer.logger.log_checkpoint(trainer.models, trainer.optimizers)
        for model in trainer.models:
            trainer.logger.log_gradients(model)

        trainer.logger.step()
        print("^"*100)
        print()
print('Training finished!...')
