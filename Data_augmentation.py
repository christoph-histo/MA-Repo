import torchvision.transforms as transforms

def elastic_transform(image, alpha, sigma):
    augmented_image = transforms.ElasticTransform(alpha=alpha, sigma=sigma)(image)
    return augmented_image

def rotate(image, angle):
    augmented_image = transforms.Rotate(angle)(image)
    return augmented_image

def flip(image, flip_type):
    augmented_image = transforms.Flip(flip_type)(image)
    return augmented_image