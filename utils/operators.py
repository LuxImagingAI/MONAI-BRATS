import os
import time

import torch.cuda
from ignite.engine import engine, Events
from monai.data import Dataset, DataLoader
from monai.deploy.core import DataPath, IOType, Image, InputContext, OutputContext, ExecutionContext, Operator
from monai.deploy.operators import InferenceOperator
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import monai.deploy.core as md
from monai.transforms import LoadImage, Compose, EnsureTyped, MeanEnsembled, SqueezeDimd, AsChannelLastd, ToTensor, \
    Lambda, AsChannelLast, SaveImage
from utils.model import inference
import nibabel as nib
import numpy as np
from monai.engines import EnsembleEvaluator, trainer
from utils.transforms import ConvertToBratsClassesBasedOnMultiChannel


@md.input("image", DataPath, IOType.DISK)
@md.output("seg_image", Image, IOType.IN_MEMORY)
class MonaiSegInferenceOperatorBRATS(InferenceOperator):
    def __init__(self, pre_transforms=None, post_transforms=None ):
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.MODEL_LOCAL_PATH = os.path.join(os.getcwd(),"output/models")
        super().__init__()

    def pre_process(self, input_path: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        pass

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        t = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading TorchScript models from: {self.MODEL_LOCAL_PATH}")
        if not os.path.isdir(self.MODEL_LOCAL_PATH): raise Exception(f"Model path is not valid: {self.MODEL_LOCAL_PATH}")

        model_file_names = [f for f in os.listdir(self.MODEL_LOCAL_PATH) if os.path.isfile(os.path.join(self.MODEL_LOCAL_PATH, f)) and not f.startswith(".")]
        models = []

        for model_file in model_file_names:
            model = torch.jit.load(os.path.join(self.MODEL_LOCAL_PATH, model_file), map_location=device)
            models.append(model)

        img_dir = op_input.get().path
        if not os.path.isdir(img_dir): raise Exception(f"Image path is not valid: {img_dir}")
        image_file_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and not f.startswith(".")]
        image_file_paths = [os.path.join(img_dir, f) for f in image_file_names]
        dataset = Dataset(data=image_file_paths, transform=self.pre_transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # post transforms need to be adjusted for the EnsembleEvaluator
        post_transforms = Compose(
            [
                EnsureTyped(keys=model_file_names),
                MeanEnsembled(
                    keys=model_file_names,
                    output_key="image",
                ),
                Lambda(lambda x: x["image"]),
                self.post_transforms,
                ConvertToBratsClassesBasedOnMultiChannel(),
                #AsChannelLast(channel_dim=0),
            ]
        )

        evaluator = EnsembleEvaluator(
            device=device,
            val_data_loader=dataloader,
            networks=models,
            inferer=inference,
            postprocessing=post_transforms,
            pred_keys=model_file_names,
        )

        self.labels = []

        @evaluator.on(Events.ITERATION_COMPLETED)
        def _save_output(engine):
            self.labels.append(engine.state.output[0])

        evaluator.run()

        op_output.set(Image(self.labels), label="seg_image")
        print(f"Ensemble segmentation finished in {round(time.time()-t,2)}s")



    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        pass

    def post_process(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        pass

@md.input("seg_image", Image, IOType.IN_MEMORY)
@md.output("seg_image_path", DataPath, IOType.DISK)
class SaveAsNiftiOperator(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.working_directory = os.getcwd()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        labels = op_input.get("seg_image").asnumpy()

        for label in labels:
            SaveImage(output_dir=os.path.join(self.working_directory, "output/labels"), separate_folder=False, output_postfix="label", scale=255)(label)