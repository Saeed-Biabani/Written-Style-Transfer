from torchvision.transforms import functional as trf

class GanTransforms:
    def __init__(self, size):
        self.size = size
    
    def __resize__(self, image):
        return trf.resize(
            image,
            self.size,
            interpolation = trf.InterpolationMode.BICUBIC,
            antialias = True
        )
    
    def __normalization__(self, image):
        return (image - image.mean()) / image.std()
    
    def __rescale__(self, image):
        return image / 255.
    
    def __call__(self, image, target):
        image = self.__normalization__(self.__resize__(image))
        target = self.__normalization__(self.__resize__(target))
        
        return image, target