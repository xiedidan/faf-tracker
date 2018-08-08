import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import numpy as np

def plot_batch(batch_data):
    images, gts = batch_data
    batch_size = len(images)
    frame_count = len(images[0])

    images = images.numpy()

    plt.ion()
    f, axs = plt.subplots(batch_size, frame_count)

    for i in range(batch_size):
        for j in range(frame_count):
            image = images[i][j]
            image = np.transpose(image, (1, 2, 0))
            
            axs[i][j].imshow(image)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            
            gt = gts[i][j]
            for k, bbox in enumerate(gt):
                plot_bbox(axs[i][j], bbox)
            
    plt.tight_layout()
    plt.ioff()

    plt.show()

def plot_bbox(ax, bbox):
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=1,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)