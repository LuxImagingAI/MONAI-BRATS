import os
import time
from pathlib import Path

import torch.cuda
from ignite.engine import Events
from monai.data import Dataset, DataLoader
from monai.deploy.core import DataPath, IOType, Image, InputContext, OutputContext, ExecutionContext, Operator
from monai.deploy.exceptions import IOMappingError
from typing import Any, Dict, Tuple, Union, Collection, Optional
import monai.deploy.core as md
from monai.transforms import Compose, EnsureTyped, MeanEnsembled, Lambda, SaveImage
from torch import nn

from utils.model import inference
from monai.engines import EnsembleEvaluator
from utils.transforms import ConvertToSingleChannel

class Models(md.domain.Domain):
    def __init__(self, models: Union[Collection[nn.Module], nn.Module], read_only: bool = False, metadata: Optional[Dict] = None):
        super().__init__(metadata=metadata)
        self._models: Union[Collection[nn.Module], nn.Module] = models
        self._read_only: bool = read_only

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, val):
        if self._read_only:
            raise IOMappingError("This DataPath is read-only.")
        self._models = val

class ImagePaths(md.domain.Domain):
    def __init__(self, image_paths: Union[str, Path, Collection[str], Collection[Path]], read_only: bool = False, metadata: Optional[Dict] = None):
        super().__init__(metadata=metadata)
        self._image_paths: Union[str, Path, Collection[str], Collection[Path]] = image_paths
        self._read_only: bool = read_only

    @property
    def paths(self):
        return self._image_paths

    @paths.setter
    def paths(self, val):
        if self._read_only:
            raise IOMappingError("This DataPath is read-only.")
        self._image_paths = val

@md.input("image", DataPath, IOType.DISK)
@md.output("image", ImagePaths, IOType.DISK)
class GetImagePathsOperator(Operator):

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        # Get image path from context and check validity
        img_dir = op_input.get().path
        if not os.path.isdir(img_dir): raise Exception(f"Image path is not valid: {img_dir}")
        print(f"Loading images from: {img_dir}")

        # Collect names of image files
        image_file_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and not f.startswith(".")]
        image_file_paths = [os.path.join(img_dir, f) for f in image_file_names]

        op_output.set(ImagePaths(image_file_paths), label="image")



@md.output("model", Models, IOType.DISK)
class GetModelsOperator(Operator):

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get model path from context and check validity
        models_dir = context.models.get_model_list()[0]["path"]
        if not os.path.isdir(models_dir): raise Exception(f"Model path is not valid: {models_dir}")
        print(f"Loading TorchScript models from: {models_dir}")

        # Collect names of model files
        model_file_names = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f)) and not f.startswith(".")]
        models = {}

        # Load models
        for model_file in model_file_names:
            model = torch.jit.load(os.path.join(models_dir, model_file), map_location=device)
            models[model_file] = model

        # Store models in output
        op_output.set(Models(models), label="model")
        print(f"Loaded {len(model_file_names)} models")



@md.input("image", ImagePaths, IOType.DISK)
@md.input("model", Models, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
class MonaiSegInferenceBRATSOperator(Operator):
    # Takes model path and image path as input, loads the models and files into memory and performs segmentation

    def __init__(self, pre_transforms=None, post_transforms=None):
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        super().__init__()

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        t = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        # Get data from input
        image_file_paths = op_input.get("image").paths
        models = op_input.get("model").models

        # Create dataset and dataloader
        dataset = Dataset(data=image_file_paths, transform=self.pre_transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Post transforms need to be extended for the EnsembleEvaluator
        # The output of the EnsembleEvaluator is in dictionary format
        post_transforms = Compose(
            [
                EnsureTyped(keys=list(models.keys())),
                MeanEnsembled(
                    keys=list(models.keys()),
                    output_key="image",
                ),
                Lambda(lambda x: x["image"]), # transforms dictionary based transform to normal transform
                self.post_transforms,
                ConvertToSingleChannel(),
            ]
        )

        # The EnsembleEvaluator will output the predictions of all supplied models for each image
        evaluator = EnsembleEvaluator(
            device=device,
            val_data_loader=dataloader,
            networks=list(models.values()),
            inferer=inference,
            postprocessing=post_transforms,
            pred_keys=list(models.keys()),
        )

        self.labels = []

        @evaluator.on(Events.ITERATION_COMPLETED)
        def _save_output(engine):
            # Collect output and append to list
            self.labels.append(engine.state.output[0].to(device="cpu"))

        evaluator.run()

        # The OutputContext (op_output) is used to transfer the output to the next Operator in the chain
        op_output.set(Image(self.labels), label="seg_image")
        print(f"Ensemble segmentation of {len(self.labels)} files finished in {round(time.time()-t,2)}s")


@md.input("seg_image", Image, IOType.IN_MEMORY)
@md.output("seg_image_path", DataPath, IOType.DISK)
class SaveAsNiftiOperator(Operator):
    # Saves the labels as files
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.working_directory = os.getcwd() # the current working directory will change when compute() is called

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        # Get the output from the previous Operator (the op_output of the previous Operator
        # is the op_input of the current Operator)
        labels = op_input.get("seg_image").asnumpy()

        # Use MONAI transform SaveImage to store the files (reads filename from MetaTensor)
        for label in labels:
            SaveImage(output_dir=op_output.get().path, separate_folder=False, output_postfix="label", scale=255)(label)
