import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

x=0
for image in glob.glob('E:/shujath-academics/academics/Projects/Major Project/dataset/sitting/*.jpg'):
    x=x+1
    img=cv2.imread(image,0)
    img = cv2.Canny(img,100,200)


    cell_size = (4, 4)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 20  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)


    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img)\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                  off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                   off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    # Preview
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    bin = 5  # angle is 360 / nbins * direction
    plt.pcolor(gradients[:, :, bin])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    
    cv2.imwrite('E:/shujath-academics/academics/Projects/Major Project/dataset/output/sit/{}.jpg'.format(x),img)
    plt.show()
