#!/usr/bin/python3

"""
Implement a simple consistent hash, mapping an example set of objects
to a set of storage devices.  Each storage device is represented
by several notional "chunks" each with their own hash code.

Using a representative set of object sizes, demonstrate the device
usage imbalance that can occur.
"""

import bisect
import copy
import datetime
import hashlib
import numpy as np
import os
import unittest

try:
    import cPickle as pickle
except ImportError:
    import pickle


class DeviceMapper(object):
    """
    Represent a chunk mapping, mapping keys to a device.
    """

    def __init__(self):
        # We separate lists of hash values for chunks and devices they map to, so we can easily bisect
        self._device_count = 0

        # List of chunk hashes, formed from the hash of 'osdN:C'
        self._chunk_map = []

        # Parallel list, with the device number for each hash
        self._device_map = []

    def create(self, chunks, device_count):
        """
        Create a fresh device mapping.
        :param chunks: The number of chunks to include.
        :param device_count: The number of devices to include.
        """
        self._device_count = device_count
        self._chunk_map = []
        self._device_map = []

        # Quicker to sort once, rather than insert into sorted order
        mapping = []
        for device in range(device_count):
            for chunk in range(chunks):
                name = 'osd{}:{}'.format(device, chunk)
                hash_value = md5_hash(name)
                mapping.append((hash_value, device))
        mapping.sort()

        for hash_value, device in mapping:
            self._chunk_map.append(hash_value)
            self._device_map.append(device)

    def get_state(self):
        """
        Get a tuple of device count, chunk map, device map.
        :return: tuple(device_count, chunk_map, device_map)
        """
        return self._device_count, self._chunk_map, self._device_map

    def set_state(self, state):
        """
        Set the device count, chunk map, and device map at once.
        :param state: tuple(device_count, chunk_map, device_map)
        """
        self._device_count, self._chunk_map, self._device_map = state

    def locate_chunk(self, hash_value):
        """
        Locate the chunk a file is on.
        :param hash_value: The hashed file name.
        :return: The chunk ID.
        """
        chunk_id = bisect.bisect(self._chunk_map, hash_value)
        if chunk_id == len(self._device_map):
            chunk_id = 0
        return chunk_id

    def locate_device(self, hash_value):
        """
        Locate the device a file is on.
        :param hash_value: The hashed file name.
        :return: The device ID.
        """
        return self._device_map[self.locate_chunk(hash_value)]

    def rebalance(self, file_data, device_size, max_load_factor):
        """
        Rebalance the data on each device to bring each device's total load under the max_load_factor.
        :param file_data: list(tuple(file hash, file size)) mappings.
        :param device_size: int The size of each device.
        :param max_load_factor: The maximum load per device in range [0.0 no data, 1.0 full disk]
        :return: The total bytes moved.
        """
        # TODO: complete this method.
        raise NotImplementedError()


class DeviceMapperTests(unittest.TestCase):
    """
    Methods to generate test data and test the DeviceMapper.rebalance implementation.
    """

    _seed = 0
    _device_count = 20
    _file_count = 1000000
    _device_size = int(2e12)
    _chunk_size = int(10e9)
    _max_factor_load = 0.9
    _cache_file = 'data-{seed}.pkl'

    def test_device_mapper(self):
        """
        A basic test for the device mapper.
        :raises AssertionError: On test fail.
        """
        file_data, mapper = self._get_data(self._seed)
        old_file_data = copy.deepcopy(file_data)
        old_mapper = copy.deepcopy(mapper)

        total_device_size = self._device_count * self._device_size
        total_file_size = float(np.sum([file_size for (hash_value, file_size) in file_data]))

        print('Files: {:.2f}T'.format(float(total_file_size) / (1 << 40)))
        print('Devices: {:.2f}T'.format(total_device_size / (1 << 40)))
        print('Use: {:.3f}%'.format(100.0 * total_file_size / total_device_size))

        print('Device usage before rebalance:')
        self._print_device_usage(file_data, mapper)

        start = datetime.datetime.now()
        bytes_moved = mapper.rebalance(file_data, self._device_size, self._max_factor_load)
        end = datetime.datetime.now()
        print('Total data movement: {:.2f}G'.format(bytes_moved / float(1 << 30)))
        print('Time taken: {}'.format(end - start))

        print('Device usage after rebalance:')
        self._print_device_usage(file_data, mapper)
        self._post_rebalance_checks(old_file_data, old_mapper, file_data, mapper, bytes_moved)

    def _post_rebalance_checks(self, old_file_data, old_mapper, file_data, mapper, bytes_moved):
        """
        Check the resulting rebalance is valid.
        :param old_file_data: The old file data.
        :param old_mapper: The old device mapper.
        :param file_data: The current file data.
        :param mapper: The current device mapper.
        :param bytes_moved: The number of bytes moved.
        :raises AssertionError: On test fail.
        """
        print('Running tests...')

        # 0. Check that file_data is unchanged.
        assert sorted(old_file_data) == sorted(file_data), 'File data should not have changed.'
        assert old_mapper.get_state() != mapper.get_state()

        # Compute device usage before and after rebalance.
        old_device_usages = np.zeros(self._device_count, dtype=np.int64)
        device_usages = np.zeros(self._device_count, dtype=np.int64)
        for hash_value, file_size in file_data:
            old_device_usages[old_mapper.locate_device(hash_value)] += file_size
            device_usages[mapper.locate_device(hash_value)] += file_size
        assert sum(old_device_usages) == sum(device_usages), 'Files/chunks have been lost!'

        # 1. Check that the devices are balanced now.
        assert all([du <= self._device_size * self._max_factor_load for du in device_usages]), \
            'Not all devices are balanced under the max factor load.'

        # 2. Check the amount of data moved.
        real_bytes_moved = sum(
            file_size for hash_value, file_size in file_data
            if old_mapper.locate_device(hash_value) != mapper.locate_device(hash_value))

        theoretical_min = self._theoretical_min_movement(old_file_data, old_mapper)
        limit = 2 * theoretical_min
        print('Data movement theoretical min: {:.2f}G, actual: {:.2f}G, max: {:.2f}G'.format(
            theoretical_min / float(1 << 30),
            real_bytes_moved / float(1 << 30),
            limit / float(1 << 30)))
        assert real_bytes_moved <= bytes_moved and real_bytes_moved <= limit, 'Bytes moved over theoretical limit.'

        # 3. Check if the hash ring and chunk-to-device mapping are valid.
        assert mapper._chunk_map == sorted(mapper._chunk_map), 'Chunk map not valid.'
        assert len(mapper._chunk_map) == len(mapper._device_map), 'Device map not valid.'

        # 4. Check that no new devices have been added.
        assert 0 <= min(mapper._device_map) <= max(mapper._device_map) <= self._device_count, 'New devices were added.'

        print('Tests passed.')

    def _theoretical_min_movement(self, file_data, mapper):
        """
        Calculate the theoretical minimum data movement.
        :param file_data: The file data.
        :param mapper: The device mapper.
        :return: The min theoretical movement between devices.
        """
        # TODO: calculate the minimum movement given the file data and original chunk/device mappings.
        raise NotImplementedError()

    def _get_data(self, seed):
        """
        Get the dataset for a particular seed.
        :param seed: The seed the dataset is generated with.
        :return: tuple(file_data, mapper).
        """
        cache_file = self._cache_file.format(seed=seed)
        if not os.path.exists(cache_file):
            print('State file {} doesn\'t exist, creating and saving an initial data set.'.format(cache_file))
            file_data, mapper = self._create_data(seed)
            self._save_data(cache_file, file_data, mapper)
        else:
            print('Loading data set {}'.format(cache_file))
            file_data, mapper = self._load_data(cache_file)
        return file_data, mapper

    def _create_data(self, seed):
        """
        Create a representative random dataset and mapping.
        :param seed: An integer PRNG seed.
        :return: tuple(list(tuple(file_hash, file_size)), DeviceMapper)
        """
        # Repeatable
        np.random.seed(seed)

        # Create a normally distributed set of files sizes, clipped to 1B -> 100GB
        file_sizes = np.clip(np.exp(3.3 * np.random.randn(self._file_count) + 12), 1, 100 * 1 << 30).astype(np.int64)

        # Calculate file name hashes based on fictional names
        file_name_hashes = [md5_hash('file-{}'.format(file_number)) for file_number in range(self._file_count)]

        mapper = DeviceMapper()
        mapper.create(self._device_size // self._chunk_size, self._device_count)

        return list(zip(file_name_hashes, file_sizes)), mapper

    def _save_data(self, file_name, file_data, mapper):
        """
        Save the random generated dataset to disk to avoid generating it every time.
        :param file_name: The name of the data file (data.pkl).
        :param file_data: The file mappings.
        :param mapper: The mapper state to save.
        """
        with open(file_name, 'wb') as file:
            pickle.dump((file_data, mapper.get_state()), file, pickle.HIGHEST_PROTOCOL)

    def _load_data(self, file_name):
        """
        Load the dataset cache from disk.
        :param file_name: The name of the data file (data.pkl)
        :return: The file data and initial mapper state.
        """
        with open(file_name, 'rb') as file:
            file_data, mapper_state = pickle.load(file)
        mapper = DeviceMapper()
        mapper.set_state(mapper_state)
        return file_data, mapper

    def _print_device_usage(self, file_data, mapper):
        """
        Print the device usage to console.
        :param mapper: The device mapper.
        :param file_data: The file data.
        """
        used = np.zeros(self._device_count, dtype=np.int64)

        for hash_value, file_size in file_data:
            used[mapper.locate_device(hash_value)] += file_size

        for device in range(self._device_count):
            print('{:2d} {:7.2f}G : {:4.1f}%'.format(
                device,
                float(used[device]) / (1 << 30),
                100.0 * float(used[device]) / self._device_size))


def md5_hash(string):
    """
    Compute the 128-bit MD5 hash for a string.
    :param string: The string to hash.
    :return: The int value of the hash.
    """
    return int(hashlib.md5(string.encode()).hexdigest(), 16)


def main():
    """
    Run the unittests for DeviceMapper.
    """
    tests = DeviceMapperTests('test_device_mapper')
    tests.test_device_mapper()


if __name__ == '__main__':
    main()
