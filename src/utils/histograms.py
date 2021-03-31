import numpy as np
import matplotlib.pyplot as plt


def bgr_histograms(image, cumulative=True):
    start = 0
    end = image.max().astype(np.uint16) + 1
    hists = [np.histogram(image[:, :, i],
                          bins=end,
                          range=(start, end),
                          density=True)[0] for i in range(3)]

    if cumulative:
        hists = list(map(np.cumsum, hists))

    fig = plt.figure()
    for i, channel in enumerate(['b', 'g', 'r']):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.plot(hists[i], channel)
    return fig
