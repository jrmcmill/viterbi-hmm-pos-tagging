"""
Run model instance
"""
from viterbi import ViterbiPOS


TRAIN_DATA = 'training.data'
TEST_DATA = 'testing.data'
SMALL_DATA = 'small_training.data'


if __name__ == '__main__':
    # initialize HMM
    model = ViterbiPOS()

    # train HMM
    print('training...')
    model.train(TRAIN_DATA)
    print('finished training\n')

    # test HMM
    print('testing...')
    model.test(TEST_DATA)
    print('finished testing\n')

