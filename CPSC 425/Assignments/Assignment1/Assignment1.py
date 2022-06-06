# -*- coding: utf-8 -*-
# CPSC 425: Assignment 1
# Name: Terence Chen
# Std number: 4260216

from PIL import Image
import numpy as np
import math
from scipy import signal

# PART 1: Gaussian Filtering 
# Question 1
def boxfilter(n):
    
    # Checks that n is odd
    assert n % 2 == 1, "Dimension must be odd" 
    
    # Ensures filter sums to 1
    return np.ones((n, n))/(n*n)
    
# Question 2 
def gauss1d(sigma):
    
    # Calulates the length and ensures it is a double
    filter_length = (math.ceil(6 * float(sigma))) 
    
    # Add one if even to ensure it is odd
    if filter_length % 2 == 0:
        filter_length += 1 
    
    # Calculate mid point and edges,sorted into array 
    k = math.floor(filter_length/2)
    d = np.arange((-1 * k), k + 1)
    
    #apply gaussian function
    gauss_fun = np.exp( -(d ** 2)/(2 * sigma ** 2))
    
    # Normalize values and return results
    return (gauss_fun / np.sum(gauss_fun)) 
    
# Question 3 
def gauss2d(sigma):
    
    # 1d to 2d, gauss1d already normalized
    two_dim = gauss1d(sigma)[np.newaxis]
    
    two_dim_transpose = np.transpose(two_dim)
    return signal.convolve2d(two_dim, two_dim_transpose)
    
# Question 4 
#Part a
def gaussconvolve2d(array,sigma):
    gc2d_filter = gauss2d(sigma);
    result = signal.convolve2d(array, gc2d_filter, 'same')
    return result

# Though convolution and correlation may seem similar for some certain examples, they ultimately 
# have different applications. Convolution is associative whereas correlation is not. Thus we need 
# two different functions ‘signal.convolve2d’ and ‘signal.correlate2d’ in Scipy.    
       
#Part b
image_file = "/Users/Terence/Desktop/dog.jpg" 

#convert image to grey scale, apply numpy array and calls gaussconvolve2d function 
image = Image.open(image_file).convert('L')
array = np.asarray(image)
convolve = gaussconvolve2d(array, 3)
 
#Part c

# Original image converted to greyscale
image.show()

# Filtered image

# Filters images using the gaussconvolve2d function
filtered_img = Image.fromarray(np.uint8(convolve))
filtered_img.show()
    
# Question 5
# We know that the Gaussian blur is separable thus by taking advantage seperability and applying Fourier transforms 
# to the functions we are efficiently simplify the process with multiplication. To further elaborate, by seperating 
# Gaussian blur function into two functions,f(x) and f(y). Then by applying Fourier transform to the functions we are able to
# efficiently reduce the convolution procedure to multiplication. Thus making the whole process more efficient. 

 
# PART 2: Hybrid Images
# Question 1      
dog_image_file = "/Users/Terence/Desktop/hw1/0b_dog.bmp" 
dog = Image.open(dog_image_file)
d_array = np.asarray(dog)

# Convolve the each rgb array with sigma = 5
red = gaussconvolve2d(d_array[:,:,0], 5)
green = gaussconvolve2d(d_array[:,:,1], 5)
blue = gaussconvolve2d(d_array[:,:,2], 5)

# Convert back to 3d after convolving
red=red[:,:,np.newaxis]
green=green[:,:,np.newaxis]
blue=blue[:,:,np.newaxis]

# Add back the arrays
filtered_dog = Image.fromarray(np.uint8(np.concatenate((red, green, blue ), axis=2)))
filtered_dog.show()
 
# Question 2 
cat_image_file = "/Users/Terence/Desktop/hw1/0a_cat.bmp" 
cat = Image.open(cat_image_file)
c_array = np.asarray(cat)

# Subtracting original with the low frequency to get high frequency, and add on 128 for visualization
c_red =  c_array[:,:,0] - gaussconvolve2d(c_array[:,:,0], 5)+128
c_green = c_array[:,:,1] - gaussconvolve2d(c_array[:,:,1], 5)+128
c_blue =  c_array[:,:,2] - gaussconvolve2d(c_array[:,:,2], 5)+128
c_red = c_red[:,:,np.newaxis]
c_green = c_green[:,:,np.newaxis]
c_blue = c_blue[:,:,np.newaxis]
filtered_cat = Image.fromarray(np.uint8(np.concatenate((c_red, c_green, c_blue ), axis=2)))
filtered_cat.show()

# Question 3
# new arrays without the +128 for high frequency
c_red1 =  c_array[:,:,0] - gaussconvolve2d(c_array[:,:,0], 5)
c_green1 = c_array[:,:,1] - gaussconvolve2d(c_array[:,:,1], 5)
c_blue1 =  c_array[:,:,2] - gaussconvolve2d(c_array[:,:,2], 5)
c_red1 = c_red1[:,:,np.newaxis]
c_green1 = c_green1[:,:,np.newaxis]
c_blue1 = c_blue1[:,:,np.newaxis] 

# adding the low and high frequency images and clipping 
filtered_combined = Image.fromarray(np.uint8(np.concatenate((np.clip((c_red1 + red),0,255), np.clip(c_green1 + green, 0 ,255), np.clip(c_blue1 + blue,0,255)), axis=2)))
filtered_combined.show()

###########################################################
#Second pair of images: fish and submarine with sigma 8
fish_image_file = "/Users/Terence/Desktop/hw1/3a_fish.bmp" 
fish = Image.open(fish_image_file)
f_array = np.asarray(fish)
f_red = gaussconvolve2d(f_array[:,:,0], 8)
f_green = gaussconvolve2d(f_array[:,:,1], 8)
f_blue = gaussconvolve2d(f_array[:,:,2], 8)
f_red=f_red[:,:,np.newaxis]
f_green=f_green[:,:,np.newaxis]
f_blue=f_blue[:,:,np.newaxis]
filtered_fish = Image.fromarray(np.uint8(np.concatenate((f_red, f_green, f_blue ), axis=2)))
filtered_fish.show()
 
#Question 2 
sub_image_file = "/Users/Terence/Desktop/hw1/3b_submarine.bmp" 
sub = Image.open(sub_image_file)
s_array = np.asarray(sub)
s_red =  s_array[:,:,0] - gaussconvolve2d(s_array[:,:,0], 8)+128
s_green = s_array[:,:,1] - gaussconvolve2d(s_array[:,:,1], 8)+128
s_blue =  s_array[:,:,2] - gaussconvolve2d(s_array[:,:,2], 8)+128
s_red = s_red[:,:,np.newaxis]
s_green = s_green[:,:,np.newaxis]
s_blue = s_blue[:,:,np.newaxis]
filtered_sub = Image.fromarray(np.uint8(np.concatenate((s_red, s_green, s_blue ), axis=2)))
filtered_sub.show()

#Question 3
# new arrays without the +128 for high frequency
s_red1 =  s_array[:,:,0] - gaussconvolve2d(s_array[:,:,0], 8)
s_green1 = s_array[:,:,1] - gaussconvolve2d(s_array[:,:,1], 8)
s_blue1 =  s_array[:,:,2] - gaussconvolve2d(s_array[:,:,2], 8)
s_red1 = s_red1[:,:,np.newaxis]
s_green1 = s_green1[:,:,np.newaxis]
s_blue1 = s_blue1[:,:,np.newaxis] 

#adding the low and high frequency images and clipping 
filtered_combined_ss = Image.fromarray(np.uint8(np.concatenate((np.clip((s_red1 + f_red),0,255), np.clip(s_green1 + f_green, 0 ,255), np.clip(s_blue1 + f_blue,0,255)), axis=2)))
filtered_combined_ss.show()

###########################################################
#Third pair of images: bird and plane with sigma 12
bird_image_file = "/Users/Terence/Desktop/hw1/4a_bird.bmp" 
bird = Image.open(bird_image_file)
b_array = np.asarray(bird)
b_red = gaussconvolve2d(b_array[:,:,0], 12)
b_green = gaussconvolve2d(b_array[:,:,1], 12)
b_blue = gaussconvolve2d(b_array[:,:,2], 12)
b_red = b_red[:,:,np.newaxis]
b_green = b_green[:,:,np.newaxis]
b_blue = b_blue[:,:,np.newaxis]
filtered_bird = Image.fromarray(np.uint8(np.concatenate((b_red, b_green, b_blue ), axis=2)))
filtered_bird.show()
 
#Question 2 
plane_image_file = "/Users/Terence/Desktop/hw1/4b_plane.bmp" 
plane = Image.open(plane_image_file)
p_array = np.asarray(plane)
p_red =  p_array[:,:,0] - gaussconvolve2d(p_array[:,:,0], 12)+128
p_green = p_array[:,:,1] - gaussconvolve2d(p_array[:,:,1], 12)+128
p_blue =  p_array[:,:,2] - gaussconvolve2d(p_array[:,:,2], 12)+128
p_red = p_red[:,:,np.newaxis]
p_green = p_green[:,:,np.newaxis]
p_blue = p_blue[:,:,np.newaxis]
filtered_plane = Image.fromarray(np.uint8(np.concatenate((p_red, p_green, p_blue ), axis=2)))
filtered_plane.show()

#Question 3
# new arrays without the +128 for high frequency
p_red1 =  p_array[:,:,0] - gaussconvolve2d(p_array[:,:,0], 12)
p_green1 = p_array[:,:,1] - gaussconvolve2d(p_array[:,:,1], 12)
p_blue1 =  p_array[:,:,2] - gaussconvolve2d(p_array[:,:,2], 12)
p_red1 = p_red1[:,:,np.newaxis]
p_green1 = p_green1[:,:,np.newaxis]
p_blue1 = p_blue1[:,:,np.newaxis] 

#adding the low and high frequency images and clipping 
filtered_combined_bp = Image.fromarray(np.uint8(np.concatenate((np.clip((p_red1 + b_red),0,255), np.clip(p_green1 + b_green, 0 ,255), np.clip(p_blue1 + b_blue,0,255)), axis=2)))
filtered_combined_bp.show()
 
 
 
 
 
 
 
 
 
 
 