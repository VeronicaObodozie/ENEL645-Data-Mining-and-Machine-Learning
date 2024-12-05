# Import useful modules and Functions

# Visualize each class
im_healthy = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_2.jpg', format = 'jpg')
im_multi = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_1.jpg', format = 'jpg')
im_rust = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_3.jpg', format = 'jpg')
im_scab = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg', format = 'jpg')

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(2, 2, 1)
ax.imshow(im_healthy)
ax.set_title('Healthy', fontsize = 20)

ax = fig.add_subplot(2, 2, 2)
ax.imshow(im_multi)
ax.set_title('Multiple Diseases', fontsize = 20)

ax = fig.add_subplot(2, 2, 3)
ax.imshow(im_rust)
ax.set_title('Rust', fontsize = 20)

ax = fig.add_subplot(2, 2, 4)
ax.imshow(im_scab)
ax.set_title('Scab', fontsize = 20)

#

#

#

#

# Code ref