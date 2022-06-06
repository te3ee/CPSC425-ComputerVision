from PIL import Image, ImageDraw
import numpy as np
import csv
import math

def ReadKeys(image):
    """Input an image and its associated SIFT keypoints.

    The argument image is the image file name (without an extension).
    The image is read from the PGM format file image.pgm and the
    keypoints are read from the file image.key.

    ReadKeys returns the following 3 arguments:

    image: the image (in PIL 'RGB' format)

    keypoints: K-by-4 array, in which each row has the 4 values specifying
    a keypoint (row, column, scale, orientation).  The orientation
    is in the range [-PI, PI] radians.

    descriptors: a K-by-128 array, where each row gives a descriptor
    for one of the K keypoints.  The descriptor is a 1D array of 128
    values with unit length.
    """
    im = Image.open(image+'.pgm').convert('RGB')
    keypoints = []
    descriptors = []
    first = True
    with open(image+'.key','rb') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC,skipinitialspace = True)
        descriptor = []
        for row in reader:
            if len(row) == 2:
                assert first, "Invalid keypoint file header."
                assert row[1] == 128, "Invalid keypoint descriptor length in header (should be 128)."
                count = row[0]
                first = False
            if len(row) == 4:
                keypoints.append(np.array(row))
            if len(row) == 20:
                descriptor += row
            if len(row) == 8:
                descriptor += row
                assert len(descriptor) == 128, "Keypoint descriptor length invalid (should be 128)."
                #normalize the key to unit length
                descriptor = np.array(descriptor)
                descriptor = descriptor / math.sqrt(np.sum(np.power(descriptor,2)))
                descriptors.append(descriptor)
                descriptor = []
    assert len(keypoints) == count, "Incorrect total number of keypoints read."
    print "Number of keypoints read:", int(count)
    return [im,keypoints,descriptors]

def AppendImages(im1, im2):
    """Create a new image that appends two images side-by-side.

    The arguments, im1 and im2, are PIL images of type RGB
    """
    im1cols, im1rows = im1.size
    im2cols, im2rows = im2.size
    im3 = Image.new('RGB', (im1cols+im2cols, max(im1rows,im2rows)))
    im3.paste(im1,(0,0))
    im3.paste(im2,(im1cols,0))
    return im3

def DisplayMatches(im1, im2, matched_pairs):
    """Display matches on a new image with the two input images placed side by side.

    Arguments:
     im1           1st image (in PIL 'RGB' format)
     im2           2nd image (in PIL 'RGB' format)
     matched_pairs list of matching keypoints, im1 to im2

    Displays and returns a newly created image (in PIL 'RGB' format)
    """
    im3 = AppendImages(im1,im2)
    offset = im1.size[0]
    draw = ImageDraw.Draw(im3)
    for match in matched_pairs:
        draw.line((match[0][1], match[0][0], offset+match[1][1], match[1][0]),fill="red",width=2)
    im3.show()
    return im3

def match(image1,image2):
    """Input two images and their associated SIFT keypoints.
    Display lines connecting the first 5 keypoints from each image.
    Note: These 5 are not correct matches, just randomly chosen points.

    The arguments image1 and image2 are file names without file extensions.

    Returns the number of matches displayed.

    Example: match('scene','book')
    """
    im1, keypoints1, descriptors1 = ReadKeys(image1)
    im2, keypoints2, descriptors2 = ReadKeys(image2)
    #
    # REPLACE THIS CODE WITH YOUR SOLUTION (ASSIGNMENT 5, QUESTION 3)
    #
    #Generate five random matches (for testing purposes)
    matched_pairs = []
    threshold_value = 0.60
    
    for i in range(len(descriptors1)):
        
        # initialize an array to store our computed angles
        angle_array = []
       
        for x in range(len(descriptors2)):
        
            # Calculate the angle between each corresponding pair of descriptors from descriptor1 and descriptor2
            angle_array.append(math.acos(np.dot(descriptors1[i], descriptors2[x])))
        
        # Sort the stored angles and get the smallest one
        min_match_angle = sorted(angle_array)[0]
        
        # The second smallest angle 
        second_min_angle = sorted(angle_array)[1]
        
        # Calculate the threshold ratio
        threshold_ratio = min_match_angle/second_min_angle
        
        # Checks that the threshold raio is sastified
        if (threshold_ratio < threshold_value):
            
            # Get the key points
            kp1 = keypoints1[i]
            kp2 = keypoints2[angle_array.index(min_match_angle)]
            
            # Add the pair of keypoints into our match pair list
            matched_pairs.append([kp1,kp2])
    
    # Question 4
    # Intializes ransac set, orientation, scale and random selection of 10 times
    ransacSet = []
    orientation_val = 55
    scale_val = 0.15
    numRandSelection = 10
    
    for i in range(numRandSelection):
        sastifiedSet = []
        
        # Apply randomization
        rand_matched_pairs = matched_pairs[np.random.randint(len(matched_pairs))]
        
        # Compute the change of scale and change of orientation for the first matched pair
        # Ensures orientation is not greater than 2pi by mod pi*2 to the computed orientation change
        scale_change1 = rand_matched_pairs[0][2] / rand_matched_pairs[1][2]
        orientation1_change = (rand_matched_pairs[0][3] - rand_matched_pairs[1][3]) % (math.pi * 2) 
        
        for pair in matched_pairs:
            
            # Compute the change of scale and change of orientation for the first matched pair
            scale_change2 = pair[0][2]/ pair[1][2]
            orientation2_change = (pair[0][3] - pair[1][3]) % (math.pi * 2)
            
            # Get the difference in scale and orientation between the two matched pairs
            scale_ratio_difference = abs(scale_change1 - scale_change2)
            orientation_difference = (orientation1_change - orientation2_change) % (math.pi * 2) 
            
            
            # Deal with the case when the difference in angle between the two pairs is greater than pi  
            if (orientation_difference > math.pi):
                
                # Subtract pi if the difference in orientation is greater than pi
                orientation_difference = orientation_difference - math.pi 
            
            # Ensures the thresholds are sastified 
            # Convert orientation from degree to radians 
            if ( scale_ratio_difference <= scale_val and orientation_difference  <= orientation_val * (math.pi/180)):
                
                # Add it to our set if threshold sastified 
                sastifiedSet.append(pair)
        
        # We want the largest set, thus we set the ransac set as the largest set
        if(len(ransacSet) < len(sastifiedSet)):
            ransacSet = sastifiedSet
    
    # Displays our new Ransac set 
    im3 = DisplayMatches(im1, im2, ransacSet)
    #
    # END OF SECTION OF CODE TO REPLACE
    #
    # im3 = DisplayMatches(im1, im2, matched_pairs)
    
    return im3

#Test run...
match('library','library2')

