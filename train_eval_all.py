import Pretrained_Architectures as models
import gc
import torch


def clean_all():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

models.Med3D_Aggregator_setup(mode="train", augmentation="no_aug",finetuned=True)
clean_all()
models.Med3D_Aggregator_setup(mode="eval", augmentation="no_aug",finetuned=True)
clean_all()

models.ResNet_3D_Aggregator_setup(mode="train", augmentation="tripath",finetuned=True)
clean_all()
models.ResNet_3D_Aggregator_setup(mode="eval", augmentation="tripath",finetuned=True)
clean_all()

models.SwinTransformer_2D_Aggregator_setup(mode="train", augmentation="tripath")
clean_all()
models.SwinTransformer_2D_Aggregator_setup(mode="eval", augmentation="tripath")
clean_all()

models.ResNet_2D_Aggregator_setup(mode="train", augmentation="tripath")
clean_all()
models.ResNet_2D_Aggregator_setup(mode="eval", augmentation="tripath")
clean_all()




