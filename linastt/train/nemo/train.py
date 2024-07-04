import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from train_utils import get_base_model, setup_dataloaders, check_vocabulary            
            
@hydra_runner(config_path="yamls", config_name="finetuning.yaml")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if hasattr(cfg, 'init_from_ptl_ckpt') and cfg.init_from_ptl_ckpt is not None:
        raise NotImplementedError(
            "Currently for simplicity of single script for all model types, we only support `init_from_nemo_model` and `init_from_pretrained_model`"
        )

    asr_model = get_base_model(trainer, cfg)
    print()
    print(type(asr_model))
    print()
    freeze_encoder = False
    if freeze_encoder:
        asr_model.encoder.freeze()
        # asr_model.encoder.apply(enable_bn_se)
        logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
    else:
        asr_model.encoder.unfreeze()
        logging.info("Model encoder has been un-frozen")
    
    # Check vocabulary type and update if needed
    asr_model = check_vocabulary(asr_model, cfg)

    # Setup Data
    asr_model = setup_dataloaders(asr_model, cfg)

    # Setup Optimizer
    asr_model.setup_optimization(cfg.model.optim)

    # Setup SpecAug
    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(asr_model)



if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter