'''
Reference : 
        1. https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
        2. https://torchmetrics.readthedocs.io/en/stable/image/inception_score.html
    
'''
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# Fréchet inception distance : quality of generated images
class FID(nn.Module):
    def __init__(self, feature=2048):
        super().__init__()
        self.fid = FrechetInceptionDistance(feature=feature)
    
    def forward(self, real_images, fake_images, grayscale=False):
        if grayscale:
            real_images = real_images.expand(-1, 3,*real_images.shape[2:])
            fake_images = fake_images.expand(-1, 3,*fake_images.shape[2:])

        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)

        val = self.fid.compute()

        self.fid.reset()

        return val


# Inception Score : access how realistic generated images are
class IS(nn.Module):
    def __init__(self, feature='logits_unbiased'):
        super().__init__()
        self.IS = InceptionScore(feature=feature)
    
    def forward(self, images, grayscale=False):
        if grayscale:
            images = images.expand(-1, 3,*images.shape[2:])

        self.IS.update(images)

        val = self.IS.compute()

        self.IS.reset()

        return val
