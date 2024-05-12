from nuplan.planning.training.preprocessing.features.trajectory import Trajectory, PDMTrajectory
from nuplan.planning.training.preprocessing.features.pyg_feature import HiVTModelFeature
# from nuplan.planning.training.modeling.models.hivt_model import HiVT
from model.hivt_model import HiVT
from model.trainer import LightningTrainer
from data.dataset import *
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.datamodule import NuplanDataModule

# torch.set_float32_matmul_precision('high')


if __name__=="__main__":
    CONFIG_PATH = './config'
    CONFIG_NAME = 'training'
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME)

    dataclass = {"hivt_pyg": HiVTModelFeature, 
                 "trajectory": Trajectory,
                 "pdm_nonreactive_trajectory":PDMTrajectory}
    data_root = "/data2/nuplan_data/"
    train_path = "data/cache_train_pdm.txt"
    val_path = "data/cache_val_pdm.txt"
    train_batch_size = 32
    val_batch_size = 128
    num_workers = 4

    pl.seed_everything(2025)
    model = instantiate(cfg.model)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    logger = TensorBoardLogger(save_dir="./output",name="lightning_logs")
    datamodule = NuplanDataModule(
        data_root=data_root,train_path=train_path,val_path=val_path,dataclass=dataclass,
        train_batch_size=train_batch_size,val_batch_size=val_batch_size,num_workers=num_workers
        )
    trainer = pl.Trainer(
        gpus=1,
        check_val_every_n_epoch=2,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
        logger = logger,
        max_epochs = 32,
        limit_train_batches=12000,
        # precision=32,
        # amp_level = "O2",
        # overfit_batches = 100,
        )
    trainer.fit(model, datamodule)


