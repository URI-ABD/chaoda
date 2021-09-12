import numpy


def create_dummy_dataset():
    data = numpy.asarray([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=numpy.float64
    )
    labels = numpy.asarray([0, 0, 1], dtype=numpy.uint8)

    print(f'writing data:\n{data}')

    numpy.save('data/dummy.npy', data, allow_pickle=False, fix_imports=False)
    numpy.save('data/dummy_labels.npy', labels, allow_pickle=False, fix_imports=False)

    return


if __name__ == '__main__':
    create_dummy_dataset()
