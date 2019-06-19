import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import setting


def plot_loss(all_losses):
    plt.figure()
    plt.plot(all_losses)


def plot_matrix(n_categories, confusion):
    all_categories = ['N', 'B']
    matplotlib.rcParams.update({'font.size': 18})

    ratios = np.array([[0.0, 0.0], [0.0, 0.0]])

    # Normalize by dividing every row by its sum
    sum = np.sum(confusion, axis=1)
    for i in range(n_categories):
        for j in range(n_categories):
            ratios[i, j] = confusion[i][j] * 1.0 / sum[i]

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(ratios)
    fig.colorbar(cax)

    ax.text(0, 0, 'TN: %d' % confusion[0][0], ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.text(1, 0, 'FP: %d' % confusion[0][1], ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.text(0, 1, 'FN: %d' % confusion[1][0], ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.text(1, 1, 'TP: %d' % confusion[1][1], ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


if __name__ == '__main__':
    confusion = [[0, 0], [0, 0]]
    confusion[0][0] = 3936
    confusion[0][1] = 1305
    confusion[1][0] = 1
    confusion[1][1] = 160
    plot_matrix(setting.n_categories, confusion)
