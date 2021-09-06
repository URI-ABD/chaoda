from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy
from tqdm import tqdm
from astropy.io import fits

from utils import paths

APOGEE_PATH = Path('/data/abd/sdss/stars/apo25m').resolve()
APO25M_OUT_PATH = paths.DATA_DIR.joinpath('apo25m_dr16.npy')
APO25M_METADATA_PATH = paths.DATA_DIR.joinpath('apo25m_dr16_filenames.csv')
NUM_DIMS = 8_575


def get_fits_file_paths() -> Dict[str, List[str]]:
    print('finding all fields...')

    fields_map: Dict[str, List[str]] = dict()
    for field in tqdm(list(APOGEE_PATH.iterdir())):
        files = list(
            file_name.name for file_name in field.iterdir()
            if 'apStar-r12-' in file_name.name
            or 'asStar-r12-' in file_name.name
        )
        if len(files) > 0:
            fields_map[field.name] = files

    # spectra_files = list()
    # list(map(spectra_files.extend, fields_map.values()))
    #
    # field_ids = [
    #     file_name.split('.')[0].split('-')[-1]
    #     for file_name in spectra_files
    # ]
    #
    # print(
    #     f'num_fields: {len(fields_map)}, '
    #     f'num_spectra: {len(spectra_files), len(set(spectra_files))}, '
    #     f'num_ids: {len(field_ids), len(set(field_ids))}'
    # )
    return fields_map


def extract_combined_spectra(fields_map: Dict[str, List[str]], *, test_chunk: Optional[int] = None):
    num_spectra = sum(map(len, fields_map.values()))
    apo25m_spectra = numpy.zeros(
        shape=(num_spectra, NUM_DIMS),
        dtype=numpy.float32,
    )
    indices_to_remove = set()

    with tqdm(total=num_spectra) as progress_bar:
        with open(APO25M_METADATA_PATH, 'w') as metadata_csv:
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

                with open(APO25M_METADATA_PATH, 'a') as metadata_csv:
                    metadata_csv.write(f'{field},{file}\n')

                apo25m_spectra[index, :] = numpy.mean(spectra, axis=1)

                progress_bar.update(1)

                if test_chunk is not None and index >= (test_chunk - 1):
                    break

            if test_chunk is not None and index >= (test_chunk - 1):
                indices_to_remove.update(range(test_chunk, num_spectra))
                break

    if len(indices_to_remove) > 0:
        print(f'removing {len(indices_to_remove)} bad spectra...')
        indices_to_keep = list(filter(lambda i: i not in indices_to_remove, range(num_spectra)))
        apo25m_spectra = apo25m_spectra[indices_to_keep, :]

    print(f'saving {apo25m_spectra.shape[0]} apo25m spectra...')
    numpy.save(
        file=APO25M_OUT_PATH,
        arr=apo25m_spectra,
        allow_pickle=False,
        fix_imports=False,
    )
    return


if __name__ == '__main__':
    paths.DATA_DIR.mkdir(exist_ok=True)

    extract_combined_spectra(get_fits_file_paths(), test_chunk=10_000)
