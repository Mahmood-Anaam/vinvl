from vinvl.scene_graph_benchmark.AttrRCNN import AttrRCNN
from vinvl.scene_graph_benchmark.config import sg_cfg
from vinvl.maskrcnn_benchmark.config import cfg
from vinvl.scene_graph_benchmark.wrappers.transforms import build_transforms
from vinvl.maskrcnn_benchmark.utils.miscellaneous import set_seed
from vinvl.maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from vinvl.scene_graph_benchmark.wrappers.utils import cv2Img_to_Image, encode_spatial_features

import torch
import json
import cv2
from pathlib import Path
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import numpy as np

import warnings
warnings.filterwarnings("ignore")





# Define the base path for the library
BASE_PATH = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = BASE_PATH.joinpath('sgg_configs', 'vgattr', 'vinvl_x152c4.yaml')

MODEL_DIR = BASE_PATH.joinpath("pretrained_model", "vinvl_vg_x152c4")
_MODEL_URL = "https://huggingface.co/michelecafagna26/vinvl_vg_x152c4/resolve/main/vinvl_vg_x152c4.pth"
_LABEL_URL = "https://huggingface.co/michelecafagna26/vinvl_vg_x152c4/resolve/main/VG-SGG-dicts-vgoi6-clipped.json"


class VinVLVisualBackbone(object):
    
    def __init__(self, device='cpu', config_file=None, opts=None):
        
        self.device = device if device else 'cpu'
        if self.device != 'cpu':
            num_of_gpus = torch.cuda.device_count()
            set_seed(1000, num_of_gpus)
        else:
            set_seed(1000, 0)
        
        self.opts = {
            "MODEL.WEIGHT": str(MODEL_DIR.joinpath("vinvl_vg_x152c4.pth")),
            "MODEL.ROI_HEADS.NMS_FILTER": 1,
            "MODEL.ROI_HEADS.SCORE_THRESH": 0.2,
            "TEST.IGNORE_BOX_REGRESSION": False,
            "DATASETS.LABELMAP_FILE": str(MODEL_DIR.joinpath("VG-SGG-dicts-vgoi6-clipped.json")),
            "TEST.OUTPUT_FEATURE": True
        }
        if opts:
            self.opts.update(opts)

        self.config_file = config_file or CONFIG_FILE

        cfg.set_new_allowed(True)
        cfg.merge_from_list(["MODEL.DEVICE", self.device])
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.merge_from_file(self.config_file)
        cfg.update(self.opts)
        cfg.set_new_allowed(False)
        cfg.freeze()

        if cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            self.model = AttrRCNN(cfg)
        else:
            raise ValueError(
                f"{cfg.MODEL.META_ARCHITECTURE} is not a valid MODEL.META_ARCHITECTURE; it must be 'AttrRCNN'")

        self.model.eval()
        self.model.to(self.device)

        model_path = MODEL_DIR.joinpath("vinvl_vg_x152c4.pth")
        label_path = MODEL_DIR.joinpath("VG-SGG-dicts-vgoi6-clipped.json")

        if not model_path.is_file():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._download_file(_MODEL_URL, model_path)

        if not label_path.is_file():
            self._download_file(_LABEL_URL, label_path)

        self.checkpointer = DetectronCheckpointer(cfg, self.model, save_dir="")
        self.checkpointer.load(str(model_path))

        with open(label_path, "rb") as fp:
            label_dict = json.load(fp)

        self.idx2label = {int(k): v for k, v in label_dict["idx_to_label"].items()}
        self.label2idx = {k: int(v) for k, v in label_dict["label_to_idx"].items()}

        self.transforms = build_transforms(cfg, is_train=False)
        self.cfg = cfg

    
    def _prepare_image(self, img):
        """
        Prepares an image for feature extraction.

        Args:
            img: Input image (file path, URL, PIL.Image, numpy array, or tensor).

        Returns:
            PIL.Image: The prepared PIL.Image in RGB format.
        """
        if isinstance(img, str):
            # File path or URL
            if img.startswith("http://") or img.startswith("https://"):
                response = requests.get(img)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            # PIL.Image
            img = img.convert("RGB")
        elif isinstance(img, np.ndarray):
            # NumPy array (assume RGB)
            img = Image.fromarray(img)
        elif torch.is_tensor(img):
            # Tensor (assume CHW format)
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
            img = Image.fromarray(np.uint8(img))
        else:
            raise ValueError("Unsupported image input type.")
        return img

    def __call__(self, imgs):
        """
        Extract features from images.

        Args:
            imgs: Single image or a batch of images (list or single instance).

        Returns:
            List[dict]: List of extracted features for each image.
        """
        if not isinstance(imgs, list):
            imgs = [imgs]  # Convert to batch format

        results = []
        for img in imgs:
            # Prepare the image
            img = self._prepare_image(img)

            # Apply transforms
            img_tensor, _ = self.transforms(img, target=None)
            img_tensor = img_tensor.to(self.device)

            # Perform inference
            with torch.no_grad():
                prediction = self.model(img_tensor)
                prediction = prediction[0].to("cpu")

            # Scale predictions to original image size
            img_width, img_height = img.size
            prediction = prediction.resize((img_width, img_height))

            # Extract features
            boxes = prediction.bbox.tolist()
            classes = [self.idx2label[c] for c in prediction.get_field("labels").tolist()]
            scores = prediction.get_field("scores").tolist()
            features = prediction.get_field("box_features").cpu().numpy()
            spatial_features = encode_spatial_features(features, (img_width, img_height), mode="xyxy")

            
            img_feats = torch.tensor(np.concatenate((features, spatial_features),axis=1),
                                    dtype=torch.float32).reshape((len(boxes),-1))
           

            results.append({
                "boxes": boxes,
                "classes": classes,
                "scores": scores,
                "img_feats": img_feats,
                "spatial_features": spatial_features,
            })

        return results

    @staticmethod
    def _download_file(url, destination):
        response = requests.get(url, stream=True)
        total_length = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024),
                              total=total_length // 1024, desc=f"Downloading {destination.name}"):
                f.write(chunk)
