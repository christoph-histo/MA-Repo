import Pretrained_Architectures as models

models.Classification_2D_ResNet18_setup(mode="train", pretrained=False, data_transform="whole_slices", augmentation="no_aug", dataset="whole_slices")
models.Classification_2D_ResNet18_setup(mode="eval", pretrained=False, data_transform="whole_slices", augmentation="no_aug", dataset="whole_slices")

models.Classification_2D_ResNet18_setup(mode="train", pretrained=True, data_transform=None, augmentation="tripath", dataset="slice_parts")
models.Classification_2D_ResNet18_setup(mode="eval", pretrained=True, data_transform=None, augmentation="tripath", dataset="slice_parts")

models.Classification_2D_ResNet18_setup(mode="train", pretrained=True, data_transform=None, augmentation="elastic", dataset="slice_parts")
models.Classification_2D_ResNet18_setup(mode="eval", pretrained=True, data_transform=None, augmentation="elastic", dataset="slice_parts")

models.Classification_2D_SwinTransformer_setup(mode="train", pretrained=False, data_transform="whole_slices", augmentation="no_aug", dataset="whole_slices")
models.Classification_2D_SwinTransformer_setup(mode="eval", pretrained=False, data_transform="whole_slices", augmentation="no_aug", dataset="whole_slices")

models.Classification_2D_SwinTransformer_setup(mode="train", pretrained=True, data_transform=None, augmentation="tripath", dataset="slice_parts")
models.Classification_2D_SwinTransformer_setup(mode="eval", pretrained=True, data_transform=None, augmentation="tripath", dataset="slice_parts")

models.Classification_2D_SwinTransformer_setup(mode="train", pretrained=True, data_transform=None, augmentation="elastic", dataset="slice_parts")
models.Classification_2D_SwinTransformer_setup(mode="eval", pretrained=True, data_transform=None, augmentation="elastic", dataset="slice_parts")

models.Classification_3D_ResNet18_setup(mode="train", augmentation="no_aug")
models.Classification_3D_ResNet18_setup(mode="eval", augmentation="no_aug")

models.Classification_3D_ResNet18_setup(mode="train", augmentation="tripath")
models.Classification_3D_ResNet18_setup(mode="eval", augmentation="tripath")

models.Classification_3D_ResNet18_setup(mode="train", augmentation="elastic")
models.Classification_3D_ResNet18_setup(mode="eval", augmentation="elastic")

models.Classification_3D_SwinTransformer_setup(mode="train", augmentation="tripath")
models.Classification_3D_SwinTransformer_setup(mode="eval", augmentation="tripath")

models.Classification_3D_SwinTransformer_setup(mode="train", augmentation="elastic")
models.Classification_3D_SwinTransformer_setup(mode="eval", augmentation="elastic")

models.Classification_Med3D_setup(mode="train", augmentation="no_aug")
models.Classification_Med3D_setup(mode="eval", augmentation="no_aug")

models.Classification_Med3D_setup(mode="train", augmentation="tripath")
models.Classification_Med3D_setup(mode="eval", augmentation="tripath")

models.Classification_Med3D_setup(mode="train", augmentation="elastic")
models.Classification_Med3D_setup(mode="eval", augmentation="elastic")

models.Classification_SwinUNETR_setup(mode="train", augmentation="no_aug")
models.Classification_SwinUNETR_setup(mode="eval", augmentation="no_aug")

models.Classification_SwinUNETR_setup(mode="train", augmentation="elastic")
models.Classification_SwinUNETR_setup(mode="eval", augmentation="elastic")

models.Classification_SwinUNETR_setup(mode="train", augmentation="tripath")
models.Classification_SwinUNETR_setup(mode="eval", augmentation="tripath")
