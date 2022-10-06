import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.engines import (
    SupervisedTrainer,
    SupervisedEvaluator
)
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.handlers import MetricLogger
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from ignite.engine import Events
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import os
import torch
import pickle

from transforms import train_transform, val_transform
from model import inference, model

root_dir = "data"
output_dir = "output"
os.makedirs(root_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
set_determinism(seed=0)

# here we don't cache any data in case out of memory issue
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=True,
    cache_rate=0.0,
    num_workers=4,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# create model, loss and optimizer
max_epochs = 10
val_interval = 1
VAL_AMP = True #TODO setup: model.py

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# storing model from model.py on the target device
model = model.to(device)

# defining loss function and optimizer
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# create handlers for training procedure
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

# create training procedure
trainer = SupervisedTrainer(device=device, max_epochs=max_epochs, train_data_loader=train_loader, network=model, optimizer=optimizer, loss_function=loss_function, amp=False)

metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

@trainer.on(Events.EPOCH_COMPLETED)
def _compute_score(engine):
    print(f"Epoch {engine.state.epoch}/{engine.state.max_epochs} loss: {engine.state.output[0]['loss']}", end="")

    model.eval()
    with torch.no_grad():

        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = inference(val_inputs)
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
with open(os.path.join("output","metric.pkl"), "wb") as handle:
    pickle.dump(metric_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
torch.jit.script(model).save(os.path.join(output_dir,"trained_model.zip"))



