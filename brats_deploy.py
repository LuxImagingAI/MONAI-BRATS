import monai.deploy.core as md  # 'md' stands for MONAI Deploy (or can use 'core' instead)
from monai.deploy.core import (
    Application,
    DataPath,
    ExecutionContext,
    Image,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)
from monai.transforms import AddChannel, Compose, EnsureType, ScaleIntensity

from transforms import train_transform, val_transform
from model import inference, model

class BratsClassifierOperator(Operator):

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

