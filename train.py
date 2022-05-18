import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

from scanner.data import STOICData
from scanner.classifier import STOICNet
from scanner.autoencoder import STOICAutoEncoder, STOICVarAutoEncoder
from scanner.convnets import r50, r18, rx3d

def train(batch_size, max_epochs, from_checkpoint=None, save_checkpoint=None):
    # config model
    model = rx3d()
    net_module = STOICNet(model, batch_size=batch_size, max_epochs=max_epochs)
    data_module = STOICData(
        data_path='/data/databases/stoic/', 
        seed=42, 
        split_ratio=0.8, 
        batch_size=batch_size
    )
    data_module.prepare_data()
    callbacks = [
        ModelCheckpoint(
            dirpath=save_checkpoint, 
            filename='{epoch}-{val_loss:.3f}',
            save_top_k=3, 
            monitor='val_loss'
        ),
        LearningRateMonitor("step"),
        StochasticWeightAveraging()
    ]
    trainer = Trainer(
        devices=1, 
        accelerator="gpu", 
        precision=16, # mixed precision
        max_epochs=max_epochs, 
        callbacks=callbacks,
        accumulate_grad_batches=4 # increase effective batch size
    )
    trainer.fit(net_module, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, required=True)
    parser.add_argument('--from_checkpoint', type=str, default=None)
    parser.add_argument('--save_checkpoint', type=str, default=None)
    args = parser.parse_args()
    train(
        args.batch_size, 
        args.max_epochs, 
        from_checkpoint=args.from_checkpoint, 
        save_checkpoint=args.save_checkpoint
    )