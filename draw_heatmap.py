import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = './heatmap_data.txt'

# Read the data from the text file
with open(file_path, 'r') as file:
    data = file.readlines()

# Extract width and height from the first line
width, height = map(int, data[0].strip().split())
#width = 980
#height = 545

# Convert the data values to a 1D numpy array
data_values = np.array([float(value) for value in data[1:]])
#data_values = np.array([float(value) for value in data])

# Reshape the data into a 2D numpy array
heatmap_data = data_values.reshape((height, width))

# Set the minimum and maximum values for the color scale
color_vmin = np.min(heatmap_data)
color_vmax = np.max(heatmap_data)
#color_vmin = 0
#color_vmax = 1500


# Create the heatmap with adjusted color range
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='jet', interpolation='nearest', vmin=color_vmin, vmax=color_vmax)
plt.colorbar()
plt.title('Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')

# Save the heatmap as an image file
plt.savefig('heatmap.png')



'''
nbins = color_vmax
plt.hist(data_values, bins=3000, density=False)
#plt.hist(data_values)
'''


# Show the heatmap
plt.show()
