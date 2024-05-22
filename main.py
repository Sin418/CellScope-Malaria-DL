import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

dataset,dataset_info = tfds.load("malaria",with_info=True, as_supervised=True, shuffle_files=True, split=['train'])

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1
BATCH_SIZE = 32
len_dataset = len(dataset)


def splits(dataset, TRAIN_RATIO, TEST_RATIO, VALIDATION_RATIO):
    train_size = int(TRAIN_RATIO * len_dataset)
    test_size = int(TEST_RATIO * len_dataset)
    validation_size = int(VALIDATION_RATIO * len_dataset)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size).take(test_size)
    validation_dataset = dataset.skip(train_size + test_size).take(validation_size)
    return train_dataset, test_dataset, validation_dataset


Train_dataset, Test_dataset, Validation_dataset = splits(dataset, TRAIN_RATIO, TEST_RATIO, VALIDATION_RATIO)

