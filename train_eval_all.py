import Pretrained_Architectures as models
import gc
import torch

def clean_all():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clean_all()
models.Classification_Med3D_setup(mode="train", augmentation="no_aug")
clean_all()
models.Classification_Med3D_setup(mode="eval", augmentation="no_aug")

clean_all()
models.Classification_Med3D_setup(mode="train", augmentation="tripath")
clean_all()
models.Classification_Med3D_setup(mode="eval", augmentation="tripath")

clean_all()
models.Classification_Med3D_setup(mode="train", augmentation="elastic")
clean_all()
models.Classification_Med3D_setup(mode="eval", augmentation="elastic")

clean_all()
models.Classification_SwinUNETR_setup(mode="train", augmentation="no_aug")
clean_all()
models.Classification_SwinUNETR_setup(mode="eval", augmentation="no_aug")

clean_all()
models.Classification_SwinUNETR_setup(mode="train", augmentation="elastic")
clean_all()
models.Classification_SwinUNETR_setup(mode="eval", augmentation="elastic")

clean_all()
models.Classification_SwinUNETR_setup(mode="train", augmentation="tripath")
clean_all()
models.Classification_SwinUNETR_setup(mode="eval", augmentation="tripath")