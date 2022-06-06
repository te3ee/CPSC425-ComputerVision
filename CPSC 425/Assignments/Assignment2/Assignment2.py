from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc

#Question2
def MakePyramid(image, minsize):
    # Scale factor of 1 in the first image before any scaling occurs
    scale_factor = 1
    image_list = []
    minSizeImg = min(image.size)
    # Stops the scaling once the scaled image will have a dimension less than minsize 
    while(minSizeImg * scale_factor  > minsize):
        # Scales the image according to the sacling factor of 0.75 appends the image
        image_list.append(image.resize((int(image.size[0] * scale_factor),int(image.size[1] * scale_factor)), Image.BICUBIC))
        # Scale factor of 0.75 that is multiplied each subsequent loop
        scale_factor = scale_factor * 0.75
    return image_list

#Question 3                
def ShowPyramid(pyramid):
    # Note: Rather than a horizontal stack we were told to construct the image shown as the zebra image in class. Also specified by the prof on piazza post @96
    # Initializes height and width as 0 
    height = width = 0
    # Set the offsets as 1 
    offset_x = offset_y = 1
    for img in pyramid:
        # Width will be the sum of the largest and second largest image
        width = pyramid[0].size[0] + pyramid[1].size[0]
        # Compute the sum of all the heights of the images
        height += img.size[1] 
    
    # Height will be the sum of all height minus the largest image
    height -= pyramid[0].size[1]
    
    # creates the background image with a white color  
    blankImage = Image.new("L", (width, height), "white")
    for img in pyramid:
        # If this is the first image then we simply paste the image into the canvas and saves offset value
        if (img == pyramid[0]):
            blankImage.paste(img, (offset_x, offset_y))
            offset_x = img.size[0]
        else:
            # Otherwise for each subsequent image we apply a y axis offset that is equivalent to the height of each level of image.
            blankImage.paste(img, (offset_x, offset_y))
            offset_y += img.size[1] 
             
    blankImage.show()

# Question 4    
def FindTemplate(pyramid, template, threshold):
	#desired template width
	tempWidth = 15
	#empty list to store matches above threshold
	detectedMatches = []
	#x and y size for each template
	x = template.size[0]
	y = template.size[1]
	#resizing template
	newTemp = template.resize((int(tempWidth), int(y/(x/tempWidth))), Image.BICUBIC)
	# Allow the first picture (only need to show matches for first image) to display colored box
	coloredImg = pyramid[0].convert('RGB')
	
	for image in pyramid:
		# Calculate the NCC and append the components to detectedMatches
		matchedTemp = np.where(ncc.normxcorr2D(image,newTemp) > threshold)
		detectedMatches.append(zip(matchedTemp[1],matchedTemp[0]))
	
	for imgLvl in range(len(detectedMatches)):
	    for coord in detectedMatches[imgLvl]:
	        #drawing out the boxes
	        draw = ImageDraw.Draw(coloredImg)
	        # x and y boundary coordinates of matching template, increases height width and height depending on t
	        x1 = int(coord[0]/0.75 ** imgLvl) - int(newTemp.size[0]/(2 * 0.75 ** imgLvl))
	        x2 = int(coord[0]/0.75 ** imgLvl) + int(newTemp.size[0]/(2 * 0.75 ** imgLvl))
	        y1 = int(coord[1]/0.75 ** imgLvl) - int(newTemp.size[1]/(2 * 0.75 ** imgLvl))
	        y2 = int(coord[1]/0.75 ** imgLvl) + int(newTemp.size[1]/(2 * 0.75 ** imgLvl))
	        
	        # draw out 4 lines to form a rectangle using coordinates
	        draw.line([(x1,y2),(x1,y1)], fill = "red", width = 2)
	        draw.line([(x1,y2),(x2,y2)], fill = "red", width = 2)
	        draw.line([(x1,y1),(x2,y1)], fill = "red", width = 2)
	        draw.line([(x2,y1),(x2,y2)], fill = "red", width = 2)
	       
	        del draw
	return coloredImg
	        
	    
      
# pathing for images
temp = Image.open('/Users/Terence/Desktop/face_detection_template.jpg')
family = Image.open('/Users/Terence/Desktop/hw2/faces/family.jpg')
jb = Image.open('/Users/Terence/Desktop/hw2/faces/judybats.jpg')
stu = Image.open('/Users/Terence/Desktop/hw2/faces/students.jpg')
tree = Image.open('/Users/Terence/Desktop/hw2/faces/tree.jpg')
fans = Image.open('/Users/Terence/Desktop/hw2/faces/fans.jpg')
sports = Image.open('/Users/Terence/Desktop/hw2/faces/sports.jpg')
        
        
        
        
                
