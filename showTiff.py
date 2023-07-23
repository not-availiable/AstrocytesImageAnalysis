import matplotlib.pyplot as plt
import tifffile as tf

# Read the TIFF file
img = tf.imread('Cell_2.tif')

# Assuming img is a 3D array, and the last dimension represents the color channels
# (img.shape is something like (height, width, 3))

# Extract the three channels
channel1 = img[:,:,0]
channel2 = img[:,:,1]
channel3 = img[:,:,2]
print(channel1)
print(channel3)

# Create a new figure
plt.figure()

# Create a subplot for the first channel
plt.subplot(131) # or (1, 3, 1)
plt.imshow(channel1, cmap='gray')
plt.title('Channel 1')

# Create a subplot for the second channel
plt.subplot(132) # or (1, 3, 2)
plt.imshow(channel2, cmap='gray')
plt.title('Channel 2')

# Create a subplot for the third channel
plt.subplot(133) # or (1, 3, 3)
plt.imshow(channel3, cmap='gray')
plt.title('Channel 3')

# Display the figure
plt.show()