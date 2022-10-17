import os
import time
import torch.cuda
from ignite.engine import Events
from monai.data import Dataset, DataLoader
from monai.deploy.core import DataPath, IOType, Image, InputContext, OutputContext, ExecutionContext, Operator
from monai.deploy.operators import InferenceOperator
from typing import Any, Dict, Tuple, Union
import monai.deploy.core as md
from monai.transforms import Compose, EnsureTyped, MeanEnsembled, Lambda, SaveImage
from utils.model import inference
from monai.engines import EnsembleEvaluator


@md.input("image", DataPath, IOType.DISK)
@md.output("seg_image", Image, IOType.IN_MEMORY)
class MonaiSegInferenceOperatorBRATS(InferenceOperator):
    # Takes model path and image path as input, loads the models and files into memory and performs segmentation

    def __init__(self, pre_transforms=None, post_transforms=None ):
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        super().__init__()

    def pre_process(self, input_path: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        # Must be overriden
        pass

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        t = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get model path from context and check validity
        models_dir = context.models.get_model_list()[0]["path"]
        if not os.path.isdir(models_dir): raise Exception(f"Model path is not valid: {models_dir}")
        print(f"Loading TorchScript models from: {models_dir}")

        # Collect names of model files
        model_file_names = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f)) and not f.startswith(".")]
        models = []

        # Load models
        for model_file in model_file_names:
            model = torch.jit.load(os.path.join(models_dir, model_file), map_location=device)
            models.append(model)
        print(f"Loaded {len(model_file_names)} models")

        # Get image path from context and check validity
        img_dir = op_input.get().path
        if not os.path.isdir(img_dir): raise Exception(f"Image path is not valid: {img_dir}")
        print(f"Loading images from: {img_dir}")

        # Collect names of image files
        image_file_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and not f.startswith(".")]
        image_file_paths = [os.path.join(img_dir, f) for f in image_file_names]

        # Create dataset and dataloader
        dataset = Dataset(data=image_file_paths, transform=self.pre_transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Post transforms need to be extended for the EnsembleEvaluator
        # The output of the EnsembleEvaluator is in dictionary format
        post_transforms = Compose(
            [
                EnsureTyped(keys=model_file_names),
                MeanEnsembled(
                    keys=model_file_names,
                    output_key="image",
                ),
                Lambda(lambda x: x["image"]), # transforms dictionary based transform to normal transform
                self.post_transforms,
            ]
        )

        # The EnsembleEvaluator will output the predictions of all supplied models for each image
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
            # Collect output and append to list
            self.labels.append(engine.state.output[0])

        evaluator.run()

        # The OutputContext (op_output) is used to transfer the output to the next Operator in the chain
        op_output.set(Image(self.labels), label="seg_image")
        print(f"Ensemble segmentation of {len(image_file_names)} files finished in {round(time.time()-t,2)}s")



    def predict(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        # Must be overriden
        pass

    def post_process(self, data: Any, *args, **kwargs) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        # Must be overriden
        pass

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