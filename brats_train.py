from monai.apps import DecathlonDataset, CrossValidation
from monai.engines import (
    SupervisedTrainer
)
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from ignite.engine import Events
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from monai.utils import set_determinism
import os
import torch
import pickle
import argparse

from utils.transforms import train_transform, val_transform, post_trans
from utils.model import inference, model

parser = argparse.ArgumentParser()
parser.add_argument('--nfolds', action="store", type=int, dest="nfolds", help="Define the number of folds for cross-validation")
parser.add_argument('--fold', action="store", type=int, dest="fold", help="Define the fold used vor validation")
parser.add_argument('--epochs', action="store", type=int, dest="epochs", help="Define the number of epochs for training")
args = parser.parse_args()

nfolds = args.nfolds
fold = args.fold
epochs = args.epochs

root_dir = "data"
output_dir = "output"
model_dir = "model"
os.makedirs(root_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
set_determinism(seed=0)


# here we don't cache any data in case out of memory issue
train_folds = [n for n in range(nfolds) if n != fold]
train_ds = CrossValidation(
    root_dir=root_dir,
    nfolds=nfolds,
    dataset_cls=DecathlonDataset,
    task="Task01_BrainTumour",
    section="training",
    transform=val_transform,
    download=True,
).get_dataset(folds=train_folds)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
print(f"Train dataloader with folds {train_folds} created")

val_ds = CrossValidation(
    root_dir=root_dir,
    nfolds=nfolds,
    dataset_cls=DecathlonDataset,
    task="Task01_BrainTumour",
    section="validation",
    transform=val_transform,
    download=False,
).get_dataset(folds=fold)
print(f"Validation dataloader with fold {fold} created")

# create model, loss and optimizer
val_interval = 1
VAL_AMP = True #TODO setup: model.py

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# defining loss function and optimizer
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# create handlers for training procedure
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# storing model from model.py on the target device
model = model.to(device)

trainer_arr = []
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# create training procedure
trainer = SupervisedTrainer(
    device=device,
    max_epochs=epochs,
    train_data_loader=train_loader,
    network=model, optimizer=optimizer,
    loss_function=loss_function,
    inferer=inference,
    amp=False,
)

metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

@trainer.on(Events.EPOCH_COMPLETED)
def _compute_score(engine):
    print(f"Epoch {engine.state.epoch}/{engine.state.max_epochs} loss: {engine.state.output[0]['loss']}", end=" ")

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = inference(val_inputs, network=model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric = dice_metric.aggregate().item()
        metric_values.append(metric)

        # dice_metric_batch returns 3 different values (one for each output channel)
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_values_tc.append(metric_tc)
        metric_wt = metric_batch[1].item()
        metric_values_wt.append(metric_wt)
        metric_et = metric_batch[2].item()
        metric_values_et.append(metric_et)

        dice_metric.reset()
        dice_metric_batch.reset()

        print(
            f"current mean dice: {metric:.4f}"
            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
        )

        # cosine annealing
        lr_scheduler.step()

# run the training procedure
trainer.run()

metric_dict = {
    "metric_values": metric_values,
    "metric_values_tc": metric_values_tc,
    "metric_values_wt": metric_values_wt,
    "metric_values_et": metric_values_et
}

# save model and metrics
with open(os.path.join("output",f"metric_fold_{fold}.pkl"), "wb") as handle:
    pickle.dump(metric_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
torch.jit.script(model).save(os.path.join(model_dir,f"model_fold_{fold}.ts"))



