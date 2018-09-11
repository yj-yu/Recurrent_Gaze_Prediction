import numpy as np
import sys
import os
import tensorflow as tf
import glob
import cPickle as pkl
from basic_graphs import NN_base
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #append parent directory to path
from models.gaze_grcn import GazePredictionGRCN as TheModel
from models.gaze_grcn import CONSTANTS, GRUModelConfig
import crc_input_data_seq
from util import log, override




def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def load_model(): ################ arg.gpu_fraction args vairable missing NOONONONONONO
    global model, config, session # do we need it to be global, not so neat 
    gpu_fraction = 0.48
    
    session = tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                  allow_growth=True),
        device_count={'GPU': True},        # self-testing: NO GPU, USE CPU
    ))
    log.warn('Reloading Model ...')

    # default configuration as of now
    config = GRUModelConfig()
    config.batch_size =  14
    
    config.loss_type = 'xentropy'

    config.dump(sys.stdout)

     
    # dummy
    data_sets = crc_input_data_seq.CRCDataSplits()
    data_sets.train = data_sets.test = data_sets.valid = []

    model = TheModel(session, data_sets, config)
    print model

    return model

def load_labels(folders_paths, data_set):
    """
    args:
     folders_paths: (string) path to clipsets 
     data_set: (string) train or test
    returns:
     Returns dictionary containing thea list of touples of label and clips for each categoryw
     labels['kiss'] = [(actionclip, label)]
    """

    ## Loading test files
    if data_set == 'train':
        data_set = '*_' + data_set + '*'
    elif data_set == 'test':
        data_set = '*' + data_set + '*'
    else:
        return NameError
    
    labels_dict = OrderedDict()
    labels_records = OrderedDict()
    i = 0
    files = glob.glob(folders_paths+data_set)
    files.sort()

    for text_file in files:
        class_label = i
        class_name = text_file.split('_')[0]  # "AnswerPhone_test" - > [AnswerPhonw, test]
        class_name = class_name.split('/')[-1]
        labels_records[class_name] = i 
        with open(text_file) as f:
            for line in f:
                foldername,_, label = line.split(' ')
                if label[0] == '1': # labels  == [-1\n]
                    #labels[class_name].append((foldername, class_label))
                    try:
                        labels_dict[foldername].append(class_label)
                    except:
                        labels_dict[foldername] = [class_label]
        i += 1
        label_list = []
        for name, labels in labels_dict.iteritems():
            label_list += (name,labels)

    return labels_dict,labels_records #, label_list


def load_data(model,dataset,labels):
    ## problem this is super slow.... dunno what to do to fixt it - need to keep the order so parallel wont work ..
    
    log.infov('Generating data')
    pred = model.generate(dataset, max_instances = None)
    log.infov('Done.')
    pred_gazemaps = pred['pred_gazemaps'] = np.asarray(pred['pred_gazemap_list'])
    gt_gazemaps = pred['gt_gazemaps'] = np.asarray(pred['gt_gazemap_list'])
    images = pred['images'] = np.asarray(pred['images_list'])
    clipnames = pred['clipname_list']
    c3d = pred['c3d_list'] = np.asarray(pred['c3d_list'])
    #gazemap_pred_dict = {}
    #gazemap_gt_dict = {}
    #frame_dict = {}
    #c3d = []
    class_v = []
    i = 0
    log.infov('Creating labels')
    labels_mat = np.zeros((len(pred_gazemaps),13) , dtype = np.uint8)
    for folder in clipnames:      
        for clip in folder:
            #c3d.append(c3d_dict[clip])
            labels_mat[i,np.array(labels[clip])] = 1
            # if i % 10 == 0:
            #    print(i)
            # if i != 0:
            #     c3d_vector = np.append(c3d_vector, np.array(c3d_dict[clip]))
            #     class_vector = np.append(class_vector, np.array(labels[clip]))
            #     # gazemap_pred_dict[clip] = np.append(gazemap_pred_dict[clip],pred_gazemaps[i], axis = 0)
            #     # gazemap_gt_dict[clip] = np.append(gazemap_gt_dict[clip],gt_gazemaps[i], axis = 0)
            #     # frame_dict[clip] = np.append(frame_dict[clip],images[i], axis = 0)
                
            #     # class_vector = np.append(class_vector, np.array(label[clip]))
                
            # else:
            #     c3d_vector = np.array(c3d_dict[clip])
            #     class_vector = np.array(labels[clip])
            #     # gazemap_pred_dict[clip] = pred_gazemaps[i]
            #     # frame_dict[clip] = images[i]
            #     # gazemap_gt_dict[clip] = gt_gazemaps[i]
            i +=1
    log.infov('Done.')
    
    #c3d = np.asarray(c3d)
    ## no point converting into np.float from list T.TTTT ->> have to manage whilst writing to trf record 
    #c3d = np.reshape(c3d,(-1,) + c3d_shape)

    #a1 = np.asarray(class_v[0])

    #labels_mat[:,a1.T] = 1

    return pred_gazemaps, gt_gazemaps, images, labels_mat, c3d, clipnames

def _write_to_tf_record(filename,mode, gazemap_pred, gazemap_gt, frame, labels, c3d):
    writer = tf.python_io.TFRecordWriter(filename)
    

    print("{} dataset has length {}".format("test",len(gazemap_pred)))
    
    for i in range(len(gazemap_pred)):
            

                                          
            
        # print how many images are saved every 1000 images  ---------->> use log instead!!
        if not i % 100:
            print('%s data: {}/{}'.format(i, len(gazemap_pred)) % (mode))
            sys.stdout.flush()
            
            
            
            
        # Load frame and append to list of 69
        ###### just rewrite this function here i gueesssssss 
        #frames = self.load_frame(addrs_feat[0][i])
        # Load c3d feautures   ## loads 69 c3d features at once sae with gazemaps, fixations and pupils
        
        #frame_length.append(frames.shape[0])
            
    
        #c3d = self.load_c3d(addrs_feat[1][i])
        
        
        #gazemaps, fixationmaps, pupils, original_height_list,original_width_list,T_list,t_list, r_list, c_list = self.load_gazedata(addrs_labels[i], self.frame_length[i])
    
            #assert(frames.shape[0] == c3d.shape[0])
            
        feature = { '/input/gazemaps_pred' :  _bytes_feature(tf.compat.as_bytes(gazemap_pred[i].tostring())), # type = np.float32
                    '/input/gazemaps_gt' :  _bytes_feature(tf.compat.as_bytes(gazemap_gt[i].tostring())), #type = np.float32
                    '/input/frame' :  _bytes_feature(tf.compat.as_bytes(frame[i].tostring())), #type = np.uint8
                    '/label/label' :  _bytes_feature(tf.compat.as_bytes(labels[i].tostring())), #type = np.uint8
                    '/input/c3d' : _bytes_feature(tf.compat.as_bytes(c3d[i].tostring()))} #type = np.foat32
                          
                
                
                        
        example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
        writer.write(example.SerializeToString())
            
            
            
                
    writer.close()
    sys.stdout.flush()




    
if __name__ == "__main__":
    import argparse
    
    labels_dict_val, labels_records = load_labels('/data1/amelie/Hollywood2/ClipSets/', 'test')
    labels_dict_train, labels_records = load_labels('/data1/amelie/Hollywood2/ClipSets/', 'train')
    #print(len(labels_dict_test['HandShake']))
    #print(len(labels_dict_train['HandShake']))
  #   import pdb; pdb.set_trace()
    #labels_dict_test = OrderedDict()
    ## PROBS MAKE TEST LONGER THAN 1 ....
    #test_folders = labels_dict_val.keys[-1]
    #labels_dict_test[labels_dict_val.keys[-1]] = labels_dict_val[labels_dict_val.keys[-1]]
    
    
    #import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention', action = 'store_true',default = False)
    args = parser.parse_args()
    dataset = crc_input_data_seq.read_crc_data_sets(
        98, 98, 49, 49, np.float32,
        use_cache=False,
        parallel_jobs=10,
        dataset='hollywood2',
        with_attention = args.attention,
        fixation_original_scale=False,  # WTF
 # XXX FOR FAST DEBUGGING AND TESTING
        )



    model = load_model()

    ### loading train data
    gazemap_pred, gazemap_gt, frame,labels,c3d,clipnames= load_data(model, dataset.train,labels_dict_train)
    if args.attention:
        attention = '_attention'
    else:
        attention = ''

    train = [gazemap_pred, gazemap_gt, frame,labels,c3d,clipnames]
    _write_to_tf_record("train" +  attention + ".tfrecord", "train", gazemap_pred, gazemap_gt, frame, labels, c3d)

    
    ### loading valid data
    gazemap_pred, gazemap_gt, frame,labels,c3d,clipnames= load_data(model, dataset.valid,labels_dict_val)

    valid = [gazemap_pred, gazemap_gt, frame,labels,c3d,clipnames]
    
    _write_to_tf_record("valid" + attention+ ".tfrecord", "valid", gazemap_pred, gazemap_gt, frame, labels, c3d)

    # gazemap_pred, gazemap_gt, frame,labels,c3d= load_data(model, dataset.test,labels_dict) 
    # import pdb; pdb.set_trace()
