import os

def save_config(trainer, config):
    out_root = "lightning_logs/version_{version}".format(version=trainer.logger.version)
    config_save_path = os.path.join(out_root, 'config_param.out')
    with open(config_save_path, 'w') as f:
        f.write('# -----------------------------------#\n')
        f.write('# Parameters of the training session #\n')
        f.write('# -----------------------------------#\n\n')

        # All parameters
        for key, value in config.items():
            f.write(f'{key}: {value}\n')

    f.write('# -------------------#\n')
    f.write('# Dataset parameters #\n')
    f.write('# -------------------#\n\n')

    # Dataset parameters
    f.write(f'Train region : {config["input_format"]}\n')
    f.write(f'Test region : {config["test_region"]}\n')
    f.write(f'Classification mode : {config["classif_mode"]}\n')

    # No need to explicitly close the file, it's handled by the context manager (with open)
