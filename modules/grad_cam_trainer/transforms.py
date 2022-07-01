import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as FT

# Util class to apply padding to all the images
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return FT.pad(image, padding, 0, 'constant')

transform=transforms.Compose([
    SquarePad(),

    transforms.Resize((224,224)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5)
])


val_transform = transforms.Compose([

	  SquarePad(),
		transforms.Resize(224),
	  transforms.CenterCrop(224),
		transforms.ToTensor()
		
])