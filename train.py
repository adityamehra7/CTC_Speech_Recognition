import hydra
from omegaconf import DictConfig
from lightning import LightningDataModule,LightningModule,Trainer
from lightning.pytorch import loggers as pl_loggers

@hydra.main(version_base="1.3",config_path="configs",config_name="train.yaml")
def main(cfg: DictConfig):
    data_module: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    logger = pl_loggers.TensorBoardLogger('tb_logs')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer,logger = logger)
    
    if cfg.get('train'):
        trainer.fit(model=model,datamodule=data_module)

if __name__ == "__main__":
    main()