#%%writefile app.py
from typing import Any, Union, Tuple, Dict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


from monai.deploy.core import (
    Application,
    ExecutionContext,
    InputContext,
    Operator,
    OutputContext,
)
# copy model: scp rmaser@iris-cluster:MONAI-BRATS/output/model.ts . TODO

from monai.deploy.operators import DICOMDataLoaderOperator, DICOMSeriesToVolumeOperator, \
    DICOMSegmentationWriterOperator, MonaiSegInferenceOperator, InferenceOperator
from utils.transforms import train_transform, val_transform, post_trans, test_transform, StoredImage
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext, DataPath
import monai.deploy.core as md
from monai.transforms import LoadImaged, LoadImage
from utils.operators import MonaiSegInferenceOperatorBRATS

from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, Compose

class BratsApp(Application):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def compose(self):
        #study_loader_op = DICOMDataLoaderOperator()
        #series_to_volume_op = DICOMSeriesToVolumeOperator()
        seg_inference_op = MonaiSegInferenceOperatorBRATS(roi_size=(240, 240, 160), overlap=0.5, pre_transforms=None, post_transforms=post_trans)
        seg_writer_op = DICOMSegmentationWriterOperator(["tumor core", "whole tumor", "enhancing tumor"])

        #self.add_flow(study_loader_op, series_to_volume_op)
        self.add_flow(seg_inference_op, io_map={"image": "image"})
        self.add_flow(seg_inference_op, seg_writer_op, io_map={"seg_image": "seg_image"})

if __name__ == "__main__":
    BratsApp(do_run=True)
