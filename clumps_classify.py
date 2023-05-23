import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


if __name__ == '__main__':
    clump_eig_vals_pd = pd.read_csv('data/clump_eig_vals.csv', sep='\t')

    clump_eig_vals = clump_eig_vals_pd[['e1', 'e2', 'e3']].values

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(clump_eig_vals[:, 0], clump_eig_vals[:, 1], clump_eig_vals[:, 2])
    plt.show()

    clump_eig_vals.sort(axis=1)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(clump_eig_vals[:, 0], clump_eig_vals[:, 1], clump_eig_vals[:, 2])
    plt.show()

    fig = plt.figure(figsize=[13, 4])
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1)
        if i == 0:
            ii, jj = 0, 1
            ax.plot(clump_eig_vals[:, ii], clump_eig_vals[:, jj], '.')
        if i == 1:
            ii, jj = 1, 2
            ax.plot(clump_eig_vals[:, ii], clump_eig_vals[:, jj], '.')
        if i == 2:
            ii, jj = 2, 0
            ax.plot(clump_eig_vals[:, ii], clump_eig_vals[:, jj], '.')

    plt.show()
