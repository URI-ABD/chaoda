import os

import numpy as np
from PIL import Image, ImageDraw
from pyclam import Manifold, criterion

from src.datasets import read, DATASETS
from src.reproduce import PLOTS_PATH

PLOTS_PATH = os.path.join(PLOTS_PATH, 'box-tree')

NORMALIZE = False
SUB_SAMPLE = 20_000
MIN_POINTS = 10
MAX_DEPTH = 30
SCALE = 1
OFFSET = 1 * SCALE
HEIGHT = 25 * SCALE - OFFSET
TOTAL_WIDTH = (1920 - 200) * SCALE
IMAGE_SIZE = (1920 * SCALE, 1080 * SCALE)


def draw_tree(manifold: Manifold) -> Image:
    im: Image = Image.new(mode='RGB', size=IMAGE_SIZE, color=(256, 256, 256))
    draw: ImageDraw = ImageDraw.Draw(im=im)

    x1, y1 = 100 * SCALE, 100 * SCALE
    x2, y2 = x1, y1 + HEIGHT

    max_lfd = max(max((cluster.local_fractal_dimension
                       for cluster in layer.clusters))
                  for layer in manifold.layers)
    mid_lfd = max_lfd / 2

    for depth, layer in enumerate(manifold.layers):
        clusters = list(sorted(list(layer.clusters)))

        widths = {cluster: float(np.log2(cluster.cardinality + 1)) for cluster in clusters}
        factor = TOTAL_WIDTH / sum(widths.values())
        widths = {cluster: factor * width for cluster, width in widths.items()}

        lfds = {cluster: cluster.local_fractal_dimension for cluster in clusters}

        for cluster in clusters:
            x2, y2 = x2 + widths[cluster], y2
            lfd = lfds[cluster]
            color = (mid_lfd - lfd) if lfd < mid_lfd else (lfd - mid_lfd)
            color = 128 + int(127 * color)
            fill = (0, 0, color) if lfds[cluster] < 1 else (color, 0, 0)
            draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=(255, 255, 255))

            x1 = x2 + OFFSET

        x1, y1 = 100 * SCALE, 100 * SCALE + (depth + 1) * (HEIGHT + OFFSET)
        x2, y2 = x1, y1 + HEIGHT

    return im


def main(dataset: str):
    data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
    manifold = Manifold(data, 'euclidean')
    manifold.build_tree(
        criterion.MaxDepth(MAX_DEPTH),
        criterion.MinPoints(MIN_POINTS),
    )

    im: Image = draw_tree(manifold)
    im.save(os.path.join(PLOTS_PATH, f'{dataset}.png'))
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    # _datasets = [
    #     'annthyroid'
    #     'vowels',
    #     'cardio',
    #     'thyroid',
    #     'musk',
    #     'satimage-2',
    #     'satellite',
    #     'optdigits',
    # ]
    _datasets = list(DATASETS.keys())
    [main(dataset=_d) for _d in _datasets]
