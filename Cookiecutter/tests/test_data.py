from make_dataset import mnist
import torch

def test_data_size():
    train_set, test_set = mnist()
    # test number of images in the train and test set 
    if(len(train_set) != 60000):
        raise ValueError('There should be 60000 images in the training set')
    #assert len(train_set) == 60000 
    assert len(test_set) == 10000 

    # test that number of labes equals the number of train images and test images respectively
    assert len(train_set.targets) == 60000
    assert len(test_set.targets) == 10000
    # test dimension of one tran image
    sample = next(iter(train_set))
    assert sample[0].shape == (1,28,28)
