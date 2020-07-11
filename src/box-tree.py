import logging
import os
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw
from pyclam import Manifold, criterion, Cluster

from src.datasets import read, DATASETS
from src.reproduce import PLOTS_PATH

PLOTS_PATH = os.path.join(PLOTS_PATH, 'box-tree')

NORMALIZE = False
SUB_SAMPLE = 100_000
MAX_DEPTH = 30
SCALE = 2
OFFSET = 1 * SCALE
HEIGHT = 25 * SCALE - OFFSET
TOTAL_WIDTH = (1920 - 200) * SCALE
IMAGE_SIZE = (1920 * SCALE, 1080 * SCALE)


def draw_tree(manifold: Manifold) -> Image:
    im: Image = Image.new(mode='RGB', size=IMAGE_SIZE, color=(256, 256, 256))
    draw: ImageDraw = ImageDraw.Draw(im=im)

    # Find sizes of all subtrees
    widths: Dict[Cluster, int] = dict()

    def calculate_widths(cluster: Cluster) -> int:
        if cluster not in widths:
            widths[cluster] = sum(map(calculate_widths, cluster.children)) if cluster.children else 1
        return widths[cluster]

    calculate_widths(manifold.root)

    # normalize lfd range to [0, 2]
    lfds: Dict[Cluster, float] = {
        cluster: cluster.local_fractal_dimension
        for layer in manifold.layers
        for cluster in layer.clusters
    }
    max_lfd, median_lfd = max(lfds.values()), float(np.median(list(lfds.values())))
    lfds = {cluster: 2 * lfd / max_lfd for cluster, lfd in lfds.items()}

    x1, y1 = 100 * SCALE, 100 * SCALE
    x2, y2 = x1 + TOTAL_WIDTH, y1 + HEIGHT
    rectangles: Dict[Cluster, Tuple[int, int, int, int]] = {manifold.root: (x1, y1, x2, y2)}
    for layer in manifold.layers:
        # constrain children to the rectangles of parent
        clusters: List[Cluster] = [cluster for cluster in layer.clusters if cluster.children]
        for cluster in clusters:
            # get parent's rectangle and move y-coordinates down by one layer
            x1, y1, x2, y2 = rectangles[cluster]
            y1, y2 = y1 + HEIGHT + OFFSET, y2 + HEIGHT + OFFSET

            # normalize width fractions for children
            child_widths = {child: widths[child] for child in cluster.children}
            factor = (x2 - x1) / sum(child_widths.values())
            child_widths = {child: factor * width for child, width in child_widths.items()}

            for child, width in child_widths.items():
                x2 = x1 + int(width)
                rectangles[child] = (x1, y1, x2, y2)
                x1 = x2 + OFFSET

        # draw rectangles for all clusters
        for cluster in layer.clusters:
            lfd = lfds[cluster]
            color = (1 - lfd) if lfd < 1 else (lfd - 1)
            color = 128 + int(127 * color)
            fill = (0, 0, color) if lfd < median_lfd else (color, 0, 0)
            outline = (255, 255, 255) if cluster.children else (0, 0, 0)
            draw.rectangle(xy=rectangles[cluster], fill=fill, outline=outline)

    return im


def main(dataset: str):
    logging.info(f"building manifold for {dataset}")
    data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
    min_points = 1 if len(data) < 4_000 else 4 if len(data) < 16_000 else 8 if len(data) < 64_000 else 16
    manifold = Manifold(data, 'euclidean')
    manifold.build_tree(
        criterion.MaxDepth(MAX_DEPTH),
        criterion.MinPoints(min_points),
    )

    logging.info(f"drawing box-tree for {dataset}")
    im: Image = draw_tree(manifold)
    im.save(os.path.join(PLOTS_PATH, f'{dataset}.png'))
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    _datasets = [
        'cardio',
        'musk',
        'thyroid',
        'vowels',
    ]
    # _datasets = list(DATASETS.keys())
    [main(dataset=_d) for _d in _datasets]
