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
from utils.transforms import train_transform, val_transform, post_trans, test_transform
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext, DataPath
import monai.deploy.core as md
from monai.transforms import LoadImaged, LoadImage
from utils.operators import MonaiSegInferenceOperatorBRATS, SaveAsNiftiOperator

from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, Compose

class BratsApp(Application):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def compose(self):
        #study_loader_op = DICOMDataLoaderOperator()
        #series_to_volume_op = DICOMSeriesToVolumeOperator()
        seg_inference_op = MonaiSegInferenceOperatorBRATS(pre_transforms=test_transform, post_transforms=post_trans)
        save_as_nifti_op = SaveAsNiftiOperator()
        #self.add_flow(study_loader_op, series_to_volume_op)
        self.add_flow(seg_inference_op, save_as_nifti_op)#, io_map={"seg_image": "seg_image"})

if __name__ == "__main__":
    BratsApp(do_run=True)
