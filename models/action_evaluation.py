import tensorflow as tf
import numpy as np
import glob
import sys
from action_classification import Classifier
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import log, override
### LOAD CHECKPOINT HERE

def create_standard_hparams():
    return tf.contrib.training.HParams(
        # Data
        feat_dimensions=[1024, 7, 7],
        label_dimensions=[13],
        batch_size=10,
        num_classes=13,
        #c3d_shape = [512,2,7,7 ]

        max_iter=100000,
        num_epochs=3,
        frame_width=98,
        frame_height=98,
        channels=3,
        gazemap_width=49,
        gazemap_height=49,
        saliencymap_width=49,
        saliencymap_height=49,
        learning_rate=0.002,
        dataset='h2',  # [crc, h2 crcxh2]
        use_gazemap=False  ## weight c3d feautures 
    )

def evaluate_checkpoints(model_name,checkpoint_path):
    #import pdb; pdb.set_trace()

    hparams = create_standard_hparams()
    with tf.Session() as sess:
        model  = Classifier(sess,hparams)
        metric_average = model.generate_and_evaluate(model_name,checkpoint_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=str, help='e.g NN, SVM, CONV')
    parser.add_argument('--checkpoint_path', required=True, type=str )
    #parser.add_argument('--output_dir', default = '.', type=str)
    #parser.add_argument('--use_gazemap', default = True, type=bool)
    args = parser.parse_args()
    # checkpoints_paths = glob.glob(args.checkpoint_path + '*.index')
    # #import pdb; pdb.set_trace()
    # checkpoints = []
    # for checkpoint_path in checkpoints_paths:
    #     checkpoint = checkpoint_path.split('.')[0]
    #     checkpoints.append(checkpoint)

    # sorted(checkpoints)  
    # for checkpoint in checkpoints:
    evaluate_checkpoints(args.model,args.checkpoint_path)
