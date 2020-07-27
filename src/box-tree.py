import logging
import os
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyclam import Manifold, criterion, Cluster

from src.datasets import read, DATASETS
from src.reproduce import PLOTS_PATH

PLOTS_PATH = os.path.join(PLOTS_PATH, 'box-tree')

NORMALIZE = False
SUB_SAMPLE = 100_000
MAX_DEPTH = 30
SCALE = 2
SIZE = 25
OFFSET = 1 * SCALE
HEIGHT = SIZE * SCALE - OFFSET
TOTAL_WIDTH = (1920 - 200) * SCALE
IMAGE_SIZE = (1920 * SCALE, 1080 * SCALE)


def _subtree_widths(cluster: Cluster, widths: Dict[Cluster, float]) -> float:
    if cluster not in widths:
        widths[cluster] = sum(_subtree_widths(child, widths) for child in cluster.children) + 1

    return widths[cluster]


def _cardinality_widths(cluster: Cluster, widths: Dict[Cluster, float]) -> float:
    if cluster not in widths:
        widths[cluster] = float(np.log2(cluster.cardinality + 1))
        [_cardinality_widths(child, widths) for child in cluster.children]

    return widths[cluster]


def _radii_widths(cluster: Cluster, widths: Dict[Cluster, float]) -> float:
    if cluster not in widths:
        widths[cluster] = float(np.log2(cluster.radius + 2))
        [_radii_widths(child, widths) for child in cluster.children]

    return widths[cluster]


def _lfd_colors(
        manifold: Manifold,
        draw: ImageDraw,
        rectangles: Dict[Cluster, Tuple[int, int, int, int]],
):
    # normalize lfd range to [0, 2]
    lfds: Dict[Cluster, float] = {
        cluster: cluster.local_fractal_dimension
        for layer in manifold.layers
        for cluster in layer.clusters
    }
    max_lfd = max(lfds.values())
    lfds = {cluster: 2 * lfd / max_lfd for cluster, lfd in lfds.items()}

    for layer in manifold.layers:
        for cluster in layer.clusters:
            lfd = lfds[cluster]
            color = (1 - lfd) if lfd < 1 else (lfd - 1)
            color = 100 + int(155 * color)
            fill = (0, 0, color) if lfd < 1 else (color, 0, 0)
            outline = (255, 255, 255) if cluster.children else (0, 0, 0)
            draw.rectangle(xy=rectangles[cluster], fill=fill, outline=outline)
    return


_WIDTH_MODES = {
    'cardinality': _cardinality_widths,
    'radii': _radii_widths,
    'subtree': _subtree_widths,
}

_COLOR_MODES = {
    'lfd': _lfd_colors,
}


def draw_rectangles(manifold: Manifold, width_mode: str, color_mode: str) -> Image:
    im: Image = Image.new(mode='RGB', size=IMAGE_SIZE, color=(256, 256, 256))
    draw: ImageDraw = ImageDraw.Draw(im=im)
    font = ImageFont.truetype(font='../arial.ttf', size=(SIZE - OFFSET) * SCALE)

    # Find sizes of all subtrees
    widths: Dict[Cluster, float] = dict()
    _WIDTH_MODES[width_mode](manifold.root, widths)

    x1, y1 = 100 * SCALE, 100 * SCALE
    x2, y2 = x1 + TOTAL_WIDTH, y1 + HEIGHT
    rectangles: Dict[Cluster, Tuple[int, int, int, int]] = {manifold.root: (x1, y1, x2, y2)}
    for layer in manifold.layers:
        draw.text(
            xy=(50 * SCALE, (95 + SIZE * layer.depth) * SCALE),
            text=f"{layer.depth}",
            fill=(0, 0, 0),
            font=font,
        )
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
    _COLOR_MODES[color_mode](manifold, draw, rectangles)

    return im


def draw_tree(
        dataset: str,
        width_modes: List[str],
        color_modes: List[str],
):
    for width_mode in width_modes:
        if width_mode not in _WIDTH_MODES:
            raise ValueError(f"{width_mode} is not a valid option for widths.\n"
                             f"options are: {_WIDTH_MODES.keys()}")

    logging.info(f"building manifold for {dataset}")
    data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
    min_points = 1 if len(data) < 4_000 else 4 if len(data) < 16_000 else 8 if len(data) < 64_000 else 16
    manifold = Manifold(data, 'euclidean')
    manifold.build_tree(
        criterion.MaxDepth(MAX_DEPTH),
        criterion.MinPoints(min_points),
    )

    for width_mode in width_modes:
        for color_mode in color_modes:
            logging.info(f"drawing box-tree for {dataset} by {width_mode}, {color_mode}")
            im: Image = draw_rectangles(manifold, width_mode, color_mode)
            im.save(os.path.join(PLOTS_PATH, f'{dataset}-{width_mode}-{color_mode}.png'))
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    # _datasets = list(DATASETS.keys())[:1]
    _datasets = list(DATASETS.keys())
    _width_modes = list(_WIDTH_MODES.keys())
    _color_modes = list(_COLOR_MODES.keys())
    [draw_tree(dataset=_d, width_modes=_width_modes, color_modes=_color_modes) for _d in _datasets]
