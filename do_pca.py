# A script for doing principal component anaĺysis (PCA) for a set of images

# Assumes all the given images are of the same size.

import numpy as np
from PIL import Image

np.set_printoptions(threshold=np.inf)



# Transforms the row-by-row column pixel vectors to a vertical stack of (width x height) images in a numpy array.
# PIL should be able to convert output to image with Image.fromarray()
def reshape_data_to_img_array(data, width, height):
    # Initialize the new image array as zeros
    new_img_array = np.zeros((np.shape(data)[1]*height, width))

    for i in range(np.shape(data)[1]):
        new_img_array[i*height:(i+1)*height, :] = np.reshape(data[:, i], (height, width))

    return new_img_array

# Given a data matrix data, normalizes the COLUMN vectors to have mean 0.
def normalize_col_mean(data):
    return data - np.mean(data, 0) # Well that was easy.


# Compute the principal component expansion of a set of vectors using m components.
# Returns the expansion coefficients as well as the resulting vectors
# data should be a matrix of column vectors to be expanded, eigvecs the expansion
# basis and m expansion dimension
def get_vector_expansion(data, m, eigvecs):
    if m > np.shape(eigvecs)[1]:
        raise Exception(f'Expansion dimension m={m} exceeds basis dimension d={np.shape(eigvecs)[1]}.')

    # Get the coefficients in the reduced dimension of the expansion
    coeffs = np.dot(np.conjugate(np.transpose(eigvecs[:,0:m])), data)

    # print(np.shape(coeffs)) # Debug

    expansion = np.zeros(np.shape(data), dtype='complex128')

    for i in range(np.shape(data)[1]):
        expansion[:, i] = np.sum(np.transpose(coeffs)[i,:]*eigvecs[:,0:m], 1)

    return coeffs, expansion



# Makes the PCA data easily visible by taking the real part and scaling the data matrix
# to values between 0 and 255 (columnwise)
def make_visible(data):
    temp = np.real(data)
    temp = temp - np.min(temp, 0)
    temp = temp*(255/np.max(temp,0))

    return temp





###################################################
### Get the image files and save the images as
### vectors in a matrix.
###################################################


# Get example files (handwritten digits 0-9, 32x32 png images)
directory = 'assets'
filetype = 'png'


# Data matrix
X = np.array([])

# Widths and heights of the images
width = 0
height = 0


for i in range(30):

    filename = f'number_{i}'

    img = Image.open(f'{directory}/{filename}.{filetype}')

    if i==0:
        width, height = img.size

    # Get the (NOTE: grayscale) values of the pixels in a numpy array (column vector)
    xi = np.transpose(np.mean(np.array([img.getdata()]), 2 ))


    if np.size(X)==0:
        X = xi
    else:
        X = np.hstack((X, xi))

    img.close()


# nof_images = np.shape(X)[1] # Not sure if needed.

# print(np.shape(X)) # Debug

# The zero-mean data matrix
Xnorm = normalize_col_mean(X)

# Calculate the covariance matrix
Cx = np.shape(Xnorm)[1]*np.dot(Xnorm, np.transpose(Xnorm))

# Compute the eigenvalues of the covariance matrix
eigval, eigvec = np.linalg.eig(Cx)





nof_plotted_images = 20



###########################################
### Do the PCA expansion and start drawing
###########################################

pca_dimension = 10

coeffs, expansion = get_vector_expansion(Xnorm, pca_dimension, eigvec)

renorm_expansion = make_visible(expansion[:, 0:nof_plotted_images])

expansion_img_arrray = reshape_data_to_img_array(renorm_expansion, width, height)
expansion_img = Image.fromarray(expansion_img_arrray)




# Normalize the (real parts of the) eigenvectors columnwise 
renorm_eigvec = make_visible(eigvec[:, 0:nof_plotted_images])

# print(renorm_eigvec[:,0]) # Debug

# Image containing the eigenvectors (as many as there were original images)
eigvecs_img_array = reshape_data_to_img_array(renorm_eigvec[:,0:np.shape(Xnorm)[1]], width, height)
eigvecs_img = Image.fromarray(eigvecs_img_array)

# The original image(s)
orig_img_array = reshape_data_to_img_array(X[:, 0:nof_plotted_images], width, height)
orig_img = Image.fromarray(orig_img_array)




####################################
### Create a nice comparison picture
####################################

full_img_array = np.hstack((orig_img_array, eigvecs_img_array, expansion_img_arrray))
full_img = Image.fromarray(full_img_array)


# Choose which of the pictures to plot
# new_img.show()
# eigvecs_img.show()
full_img.show()












