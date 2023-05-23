from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from skimage import io, color


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
# plt.style.use('default')


def cut_and_plot(x1=-16, x2=16, y1=-16, y2=16, rect=None):

    if rect is None:
        rect = [0.12, 0.1, 0.84, 0.896]
    path = 'data/Milky_way_with_armname.jpg'
    img = io.imread(path)
    # 图片转成灰度图
    img_gray = color.rgb2gray(img)
    # 黑白色反转:
    img_gray = 1 - img_gray
    '''
    x1,x2,y1,y2 分别是x轴和y轴的取值范围, 单位是kpc.
    x轴和y轴的最大取值范围都是[-24,24]
    '''
    # 裁剪图片
    pixel_x1 = int((1 - (y2 - y1) / 48) / 2 * img_gray.shape[1])
    pixel_x2 = int(img_gray.shape[1] - pixel_x1)
    pixel_y1 = int((1 - (x2 - x1) / 48) / 2 * img_gray.shape[0])
    pixel_y2 = int(img_gray.shape[0] - pixel_y1)

    img_gray_cut = img_gray[pixel_x1:pixel_x2 + 1, pixel_y1:pixel_y2 + 1]

    # 画图
    fig = plt.figure(figsize=(8, 7.6))
    ax = fig.add_axes(rect)
    ax.imshow(img_gray_cut, extent=[x1, x2, y1, y2], cmap='gray')
    # ax.scatter(x_1, -1 * (y_1 - 8.15), c='red', marker='.', s=5, alpha=0.6, zorder=5)
    # 标出太阳位置
    ax.scatter(0, 8.15, marker='$\odot$', s=40, c='lime', alpha=0.8, zorder=15)  # zorder控制图层
    # 标出银心位置
    ax.scatter(0, 0, marker='*', s=40, c='red', alpha=0.8, zorder=15)  # zorder控制图层

    ax.set_xlabel('x (kpc)', fontsize=15)
    ax.set_ylabel('y (kpc)', fontsize=15)
    ax.set_xticks(np.arange(x1 + abs(x1) % 5, x2 + 1, 5),  )
    ax.set_yticks(np.arange(y1 + abs(y1) % 5, y2 + 1, 5))
    ax.grid(alpha=0.1, c='gray', ls='--')
    ax.minorticks_on()
    # ax.set_axis_off()

    # 画极坐标
    polar_x = np.linspace(x1, x2, 10)
    for i in range(12):
        if (i != 6) & (i != 12):
            polar_y = np.tan(np.pi / 12 * i) * polar_x + 8.15
            plt.plot(polar_x, polar_y, c='gray', linestyle='--', alpha=0.2)
        else:
            plt.axvline(x=0, c='gray', linestyle='--', alpha=0.4)
            plt.axhline(y=0, c='gray', linestyle='--', alpha=0.4)
            plt.axhline(y=8.15, c='gray', linestyle='--', alpha=0.4)

    # 画出高、中、低典型区域
    polar_x = np.linspace(0, x2, 10)
    polar_y = np.tan(np.pi * 280 / 180) * polar_x + 8.15
    plt.plot(polar_x, polar_y, c='red', linestyle='--', alpha=0.6)
    polar_y2 = np.tan(np.pi * 290 / 180) * polar_x + 8.15
    plt.plot(polar_x, polar_y2, c='red', linestyle='--', alpha=0.6)
    plt.fill_between(polar_x, polar_y, polar_y2, facecolor="red", alpha=0.4)

    polar_x = np.linspace(0, x2, 10)
    polar_y = np.tan(np.pi * 10 / 180) * polar_x + 8.15
    plt.plot(polar_x, polar_y, c='orange', linestyle='--', alpha=0.6)
    polar_y2 = np.tan(np.pi * 20 / 180) * polar_x + 8.15
    plt.plot(polar_x, polar_y2, c='orange', linestyle='--', alpha=0.6)
    plt.fill_between(polar_x, polar_y, polar_y2, facecolor="orange", alpha=0.4)

    polar_x = np.linspace(x1, 0, 10)
    polar_y = np.tan(np.pi * 90.01 / 180) * polar_x + 8.15
    plt.plot(polar_x, polar_y, c='green', linestyle='--', alpha=0.6)
    polar_y2 = np.tan(np.pi * 105 / 180) * polar_x + 8.15
    plt.plot(polar_x, polar_y2, c='green', linestyle='--', alpha=0.6)
    plt.fill_between(polar_x, polar_y, polar_y2, facecolor="green", alpha=0.4)

    # 画同心圆,标示距离
    def plot_circle(r):
        c = plt.Circle((0, 8.15), radius=r, color='gray', alpha=0.2, fill=False, linestyle='--')
        plt.gca().add_artist(c)

    for i in np.arange(2.5, y2 + 8.15, 2.5):
        plot_circle(i)

    ax.set_xlim(x1+7, x2)
    ax.set_ylim(y1+7, y2)


# -----------函数：添加比例尺--------------
def add_scalebar(lon, lat, length):
    plt.hlines(y=lat, xmin=lon, xmax=lon + length, colors="black", ls="-", lw=1, label='%d kpc' % (length))
    plt.vlines(x=lon, ymin=lat - 0.1, ymax=lat + 0.4, colors="black", ls="-", lw=1)
    plt.vlines(x=lon + length, ymin=lat - 0.1, ymax=lat + 0.4, colors="black", ls="-", lw=1)
    plt.text(lon + length / 2, lat + 0.5, '2.5 kpc', horizontalalignment='center')


if __name__ == '__main__':

    # cut_and_plot()
    # add_scalebar(-14, -14, 2.5)

    # plt.tight_layout()
    # plt.savefig('spiral_arm/spatial_distribution_cutted.png', dpi=200)
    # plt.savefig('spiral_arm/spatial_distribution_cutted.pdf', dpi=200)
    # plt.savefig('spiral_arm/spatial_distribution_cutted.jpg', dpi=200)
    # plt.savefig('spiral_arm/spatial_distribution_cutted.eps', dpi=200)
    # plt.show()

    cut_and_plot(x1=-20, x2=20, y1=-20, y2=20)
    add_scalebar(-10, 15, 2.5)

    # plt.tight_layout()
    # plt.savefig('spiral_arm/spatial_distribution_cutted_loc.png', dpi=200)
    # plt.savefig('./spatial_distribution_cutted.pdf', dpi=200)
    plt.savefig('./spatial_distribution_cutted.png', dpi=200)
    # plt.savefig('./spatial_distribution_cutted.eps', dpi=200)
    plt.show()