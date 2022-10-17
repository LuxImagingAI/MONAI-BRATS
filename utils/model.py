from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SegResNet

VAL_AMP = False

model = SegResNet(
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)

inference = SlidingWindowInferer(
    roi_size=(240, 240, 160),
    sw_batch_size=1,
    overlap=0.5,
)