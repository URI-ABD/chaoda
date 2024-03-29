import json
from time import time

import numpy
from pyclam import CHAODA

from chaoda.benchmark_chaoda import META_MODELS
from utils import constants
from utils import helpers
from .preparse import APOGEE_OUT_PATH
from .preparse import JSON_PATH

__all__ = ['score_apogee']


def score_apogee(fast: bool = True):
    data = numpy.load(APOGEE_OUT_PATH, mmap_mode='r')

    speed_threshold = max(128, int(numpy.sqrt(data.shape[0]))) if fast else None
    print(f'speed threshold set to {speed_threshold}')

    start = time()

    detector: CHAODA = CHAODA(
        metrics=constants.METRICS,
        max_depth=constants.MAX_DEPTH,
        min_points=helpers.assign_min_points(data.shape[0]),
        meta_ml_functions=META_MODELS,
        speed_threshold=speed_threshold,
    ).fit(data=data)

    time_taken = float(time() - start)

    scores = list(map(float, detector.scores))
    json_object = json.dumps(
        obj={'time_taken': time_taken, 'scores': scores},
        indent=4,
    )

    with open(JSON_PATH, 'w') as json_file:
        json_file.write(json_object)

    return
