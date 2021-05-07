import numpy as np
import gzip
import os

data_dir = r'./data/MNIST'


class database(object):
    def __init__(self, data, label):
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        data = data.astype(np.float32)
        data = np.multiply(data, 1.0 / 255.0)
        self.label = label
        self.data = data

    def __len__(self):
        return len(self.label)


class GetDataSet(object):
    def __init__(self, dataSetName, alpha):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(alpha)
        else:
            pass

    def mnistDataSetConstruct(self, isIID):
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        train_labels = np.argmax(train_labels, axis=1)
        test_labels = np.argmax(test_labels, axis=1)
        train_mnist = database(train_images, train_labels)
        test_mnist = database(test_images, test_labels)

        if isIID == 0:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_mnist.data[order]
            self.train_label = train_mnist.label[order]
        else:
            separate_train_minst = separate_dataset(train_mnist)
            self.train_data = separate_train_minst

        self.test_data = test_mnist.data
        self.test_label = test_mnist.label


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


def separate_dataset(dataset):
    separated_dataset = []
    for number in range(10):
        separated_dataset.append(dataset.data[(dataset.label == number)])
    return separated_dataset
