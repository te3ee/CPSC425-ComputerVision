from PIL import Image
import numpy as np
imageFile = "/Users/Terence/Desktop/peacock.png"
im = Image.open(imageFile)
print im.size, im.mode, im.format
