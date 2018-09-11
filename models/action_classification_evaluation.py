import numpy as np
import tensorflow as tf
import glob
import sys
import os
from action_classification import Classifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #append parent directory to path
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=str, help='eg. NN, SVM')
    parser.add_argument('--attention', required=False,
                        action='store_true', default = False)
    parser.add_argument('--checkpoint_path', required=True, type=str)
    args = parser.parse_args()
    
    checkpoint_list = glob.glob(args.checkpoint_path + '/*.meta')
    import pdb; pdb.set_trace()
    
    for checkpoint[:-5] in checkpoint_list:
        tf.reset_default_graph()
        model = Classifier(hparams)
        with tf.Session() as sess:
        
            model.session = sess
            attention = args.attention
            model.generate_and_evaluate(args.model, attention, checkpoint)

