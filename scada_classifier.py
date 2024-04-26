import os
import clip
import time
import torch
import config
import logging
from PIL import Image
from collections import Counter
from zeroshot import zeroshot_classifier
from openai_imagenet_template import openai_imagenet_template
from timm.data.transforms_factory import transforms_imagenet_train
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr



class ScadaClassifier():

  def __init__(self, cfg):
    # init config for inference 
    # cfg = config.CLASSIFY.clone()
    # scadaclassifier = ScadaClassifier(cfg) 
      self.setup_modelsoup(cfg)

  def setup_modelsoup(self, cfg):
      
      NUM_CLASSES = 2
      DEVICE = cfg.DEVICE

      if cfg.CUSTOM_TEMPLATE:
          template = [lambda x : f"a photo of a {x}."]
      else:
          template = openai_imagenet_template

      self.base_model, self.preprocess = clip.load(cfg.MODEL, 'cpu', jit=False)
      clf = zeroshot_classifier(self.base_model, ['open', 'close'], template, DEVICE)
      feature_dim = self.base_model.visual.output_dim

      self.model = ModelWrapper(self.base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
      for p in self.model.parameters():
          p.data = p.data.float()

      self.model = self.model.to(DEVICE)
      devices = [x for x in range(torch.cuda.device_count())]
      self.model.load_state_dict(torch.load(cfg.CHECKPOINT_PATH))
      self.model = torch.nn.DataParallel(self.model,  device_ids=devices)

  def scale_coordinates(self, x1, y1, x2, y2, ratio):
    # Calculate original width and height
    width = x2 - x1
    height = y2 - y1
    
    # Calculate new width and height
    new_width = width * ratio
    new_height = height * ratio
    
    # Calculate center of original bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate new coordinates
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

  def extract_bbox_ratio(self, img, bbox):

      x1, y1, x2, y2 = bbox
      img_ori = img.crop([int(x1), int(y1), int(x2), int(y2)])
      img_ori = self.preprocess(img_ori)[None, ...]
      
      x1_25, y1_25, x2_25, y2_25 = self.scale_coordinates(x1, y1, x2, y2, 1.25)
      img_ratio_25 = img.crop([int(x1_25), int(y1_25), int(x2_25), int(y2_25)])
      img_ratio_25 = self.preprocess(img_ratio_25)[None, ...]
      
      x1_6, y1_6, x2_6, y2_6 = self.scale_coordinates(x1, y1, x2, y2, 1.6)
      img_ratio_6 = img.crop([int(x1_6), int(y1_6), int(x2_6), int(y2_6)])
      img_ratio_6 = self.preprocess(img_ratio_6)[None, ...]
      img = torch.cat((img_ori, img_ratio_25, img_ratio_6), 0)
      return img


  def classify(self, f, bbox):
      img = Image.open(f)
      img = img.convert("RGB")
      img = self.extract_bbox_ratio(img, bbox)
      logits = self.model(img)
      pred = logits.argmax(dim=1, keepdim=True)
      result = [int(pred[0]), int(pred[1]), int(pred[2])]
      result_counts = Counter(result)
      majority_result = max(result_counts, key=result_counts.get)
      return majority_result
     
  