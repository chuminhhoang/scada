from yacs.config import CfgNode as CN

CLASSIFY = CN()
CLASSIFY.CHECKPOINT_PATH = 'model/finetune_cp_19.pt'
CLASSIFY.MODEL = 'ViT-B/32'
CLASSIFY.CUSTOM_TEMPLATE = 'False'
CLASSIFY.DEVICE = 'cpu'
