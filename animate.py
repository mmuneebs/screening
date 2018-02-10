# Example plotting function: .aps file animation
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import reader
import numpy as np
import cv2
# matplotlib.rc('animation', html='html5')


def plot_image(path):
    data = reader.read_data(path)
    fig = matplotlib.pyplot.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:, :, i].transpose()), cmap='viridis')
        return [im]
    return matplotlib.animation.FuncAnimation(fig, animate, frames=data.shape[2], interval=200, blit=True)

def save_img(path):
    data = reader.read_data(path)
    for frame in xrange(data.shape[2]):
        cv2.imwrite('image' + str(frame) + '.png', data[:, :, frame])


def see(ls, ext, path):
    index = 0
    key = 0
    slc = 0
    while key != 'q':
        key = 0
        data = reader.read_data(path+ls[index]+ext)
        # (horizontal axis(?), depth axis(front->back), height(floor->ceil)
        data = np.flipud(data.transpose([1, 0, 2]))
        data /= np.max(data)
        while key not in ['q', 'n', 'p']:
            cv2.imshow('viewer', data[:, :, slc])
            key = chr(cv2.waitKey() % 256)
            if key in ['n', 'p']:
                continue
            elif key == 'b':
                slc = (slc-1+data.shape[2]) % data.shape[2]
            else:
                slc = (slc+1) % data.shape[2]
        if key == 'n':
            index += 1
        elif key == 'p':
            index -= 1
        index = max(0, min(index, len(ls)-1))

# def see_single():
#     for c in range(0, data.shape[1], 10):
#         cv2.imshow('a3d', data[:, c, :])
#         cv2.waitKey()

# ani = plot_image('/media/sf_D_DRIVE/tsa/sample/0043db5e8c819bffc15261b1f1ac5e42.aps')
# plt.show()
# save_img('/media/sf_D_DRIVE/tsa/sample/0043db5e8c819bffc15261b1f1ac5e42.aps')

# data = reader.read_data('/media/sf_D_DRIVE/tsa/sample/0043db5e8c819bffc15261b1f1ac5e42.aps')

# see_a3d('/media/sf_D_DRIVE/tsa/sample/0043db5e8c819bffc15261b1f1ac5e42.a3d', dim=1)
# see_a3d('/media/sf_D_DRIVE/tsa/sample/00360f79fd6e02781457eda48f85da90.a3d', dim=1)

with open('list_data.txt') as ll:
    ls = ll.read().splitlines()
    see(ls, '.a3daps', '/Users/muneeb/ml/data/')
