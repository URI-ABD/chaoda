from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy
from tqdm import tqdm
from astropy.io import fits

from utils import paths

# APO_CHUNK = '1m'
APO_CHUNK = '25m'

APOGEE_PATH = Path(f'/data/abd/sdss/stars/apo{APO_CHUNK}').resolve()
APOGEE_OUT_PATH = paths.DATA_DIR.joinpath(f'apo{APO_CHUNK}.npy')
APOGEE_METADATA_PATH = paths.DATA_DIR.joinpath(f'apo{APO_CHUNK}_filenames.csv')
NUM_DIMS = 8_575


def get_fits_file_paths() -> Dict[str, List[str]]:
    print('finding all fields...')

    fields_map: Dict[str, List[str]] = dict()
    for field in tqdm(list(APOGEE_PATH.iterdir())):
        files = list(
            file_name.name for file_name in field.iterdir()
            if 'apStar-' in file_name.name
            or 'asStar-' in file_name.name
        )
        if len(files) > 0:
            fields_map[field.name] = files

    num_spectra = sum(map(len, fields_map.values()))
    print(f'collected {num_spectra} matching fits files for {APO_CHUNK}...')

    return fields_map


def extract_combined_spectra(fields_map: Dict[str, List[str]], *, test_chunk: Optional[int] = None):
    num_spectra = sum(map(len, fields_map.values()))
    apogee_spectra = numpy.zeros(
        shape=(num_spectra, NUM_DIMS),
        dtype=numpy.float32,
    )
    indices_to_remove = set()

    with tqdm(total=num_spectra) as progress_bar:
        with open(APOGEE_METADATA_PATH, 'w') as metadata_csv:
            metadata_csv.write(f'field,fits_name\n')

        index = -1
        for field, files in fields_map.items():
            field_path = APOGEE_PATH.joinpath(field)

            for file in files:
                index += 1
                file_path = field_path.joinpath(file)

                # noinspection PyBroadException
                try:
                    with fits.open(str(file_path)) as hdul:
                        spectra = numpy.asarray(hdul[1].data, dtype=numpy.float32)
                except Exception as _:
                    indices_to_remove.add(index)
                    continue

                if spectra.ndim == 1:
                    spectra = numpy.expand_dims(spectra, axis=1)
                else:
                    spectra = spectra.T

                with open(APOGEE_METADATA_PATH, 'a') as metadata_csv:
                    metadata_csv.write(f'{field},{file}\n')

                apogee_spectra[index, :] = numpy.mean(spectra, axis=1)

                progress_bar.update(1)

                if test_chunk is not None and index >= (test_chunk - 1):
                    break

            if test_chunk is not None and index >= (test_chunk - 1):
                indices_to_remove.update(range(test_chunk, num_spectra))
                break

    if len(indices_to_remove) > 0:
        print(f'removing {len(indices_to_remove)} bad spectra...')
        indices_to_keep = list(filter(lambda i: i not in indices_to_remove, range(num_spectra)))
        apogee_spectra = apogee_spectra[indices_to_keep, :]

    print(f'saving {apogee_spectra.shape[0]} apo{APO_CHUNK} spectra...')
    numpy.save(
        file=APOGEE_OUT_PATH,
        arr=apogee_spectra,
        allow_pickle=False,
        fix_imports=False,
    )
    return


if __name__ == '__main__':
    paths.DATA_DIR.mkdir(exist_ok=True)

    get_fits_file_paths()
    # extract_combined_spectra(get_fits_file_paths())
    # extract_combined_spectra(get_fits_file_paths(), test_chunk=10_000)
