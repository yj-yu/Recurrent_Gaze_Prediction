#!/usr/bin/env python
import cv2
import numpy as np
import os
import sys
import subprocess
import pickle as pkl
import glob
import collections
import array
import numpy as np
import h5py
def read_binary_blob(filename):
    #
    # Read binary blob file from C3D
    # INPUT
    # filename    : input filename.
    #
    # OUTPUT
    # s           : a 1x5 matrix indicates the size of the blob
    #               which is [num channel length height width].
    # blob        : a 5-D tensor size num x channel x length x height x width
    #               containing the blob data.
    # read_status : a scalar value = 1 if sucessfully read, 0 otherwise.


    # precision is set to 'single', used by C3D

    # open file and read size and data buffer
    # [s, c] = fread(f, [1 5], 'int32');
    read_status = 1
    blob = collections.namedtuple('Blob', ['size', 'data'])

    f = open(filename, 'rb')
    s = array.array("i") # int32
    s.fromfile(f, 5)

    if len(s) == 5 :
        m = s[0]*s[1]*s[2]*s[3]*s[4]

        # [data, c] = fread(f, [1 m], precision)
        data_aux = array.array("f")
        data_aux.fromfile(f, m)
        data = np.array(data_aux.tolist())

        if len(data) != m:
            read_status = 0;

    else:
        read_status = 0;

    # If failed to read, set empty output and return
    if not read_status:
        s = []
        blob_data = []
        b = blob(s, blob_data)
        return s, b, read_status

    # reshape the data buffer to blob
    # note that MATLAB use column order, while C3D uses row-order
    # blob = zeros(s(1), s(2), s(3), s(4), s(5), Float);
    blob_data = np.zeros((s[0], s[1], s[2], s[3], s[4]), np.float32)
    off = 0
    image_size = s[3]*s[4]
    for n in range(0, s[0]):
        for c in range(0, s[1]):
            for l in range(0, s[2]):
                # print n, c, l, off, off+image_size
                tmp = data[np.array(range(off, off+image_size))];
                blob_data[n][c][l][:][:] = tmp.reshape(s[3], -1);
                off = off+image_size;


    b = blob(s, blob_data)
    f.close()
    return s, b, read_status

gpu_id = 0

batch_size = 50
__force_computing__ = True
__video_width__ = 400.0 
caffe_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../..'
        ))

def check_trained_model(trained_model):
    """Checks if the trained_model is there, otherwise it gets downloaded"""

    if os.path.isfile(trained_model):
        print("[Info] trainded_model={} found. Continuing... ".format(trained_model))
    else:
        donwload_cmd = [
            "wget",
            "-0",
            trained_model,
            "https://www.dropbox.com/s/vr8ckp0pxgbldhs/conv3d_deepnetA_sport1m_iter_1900000?dl=0",
            ]

        print("[Info] Download sports1m pre-trained model: \"{}\"".format(' '.join(cownload_cmd)
        ))

        return_code  = subprocess.call(download_cmd) # calls command in command line

        if return_code != 0:
            print("[Error] Downloading of pretrained model failed. Check!")
            sys.exit(-10)

    return


def get_frame_count(video):
    """Get frame count and FPS for a single video clip """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[Error] video={} cannot be opened.".format(video))
        sys.exit(-6) #termiante python with exit status -6

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   

    fps = cap.get(cv2.CAP_PROP_FPS) #frames per second
    if not fps or fps != fps:
        fps = 29.97

    return num_frames, fps

def extract_frames(video, start_frame, frame_dir,num_frames, num_frames_to_extract=16):
    # check output directory
    video_id = video.split('/')[-1].split('.')[0]
    frame_dir = os.path.join(frame_dir, video_id)
    if os.path.isdir(frame_dir):
        print ("[Warning] frame_dir={} does exist. Will overwrite".format(frame_dir))
    else:
        os.makedirs(frame_dir)

    cap = cv2.VideoCapture(video) #returns video_capture object
    if not cap.isOpened():
        print ("[Error] video={} can not be opened.".format(video))
        sys.exit(-6)

   # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = __video_width__ / width
    new_height = int(scale * height)
    new_width = int(scale * width)
    frames_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #get frames and save
    #for frame_count in range(0,(num_frames - num_frames_to_extract), num_frames_to_extract):
    for frame_count in range( num_frames):
        
    
        frame_num = frame_count #+ start_frame
        print("[Info] Extracting frame num={}".format(frame_num))
        ret, frame = cap.read()
        if not ret:
            print("[Error] Frame extraction was not successful")
            sys.exit(-7)
        
        
        frame = cv2.resize(frame, (new_width,new_height))
        frame_file = os.path.join(frame_dir, '{0:06d}.jpg'.format(int(frame_num) + 1))
        #if frame_num%16 ==0:
        cv2.imwrite(frame_file, frame)
    # if len(glob.glob("%s/*.jpg" % frame_dir)) != frames_n+1:
    #     print("===============", frames_n, "===============")
    #     cnt = 1
    #     while True:
    #         success, frame = cap.read()
    #         if not success:
    #             break
    #         frame = cv2.resize(frame, (new_width,new_height))
    #         cv2.imwrite("%s/%06d.jpg" % (frame_dir,cnt), frame)
    #         cnt += 1
    # cap.release()

    # return


def generate_feature_prototxt(out_file, src_file, mean_file=None): # what is mean_file? -> sport1m_train16_128_mean.binaryproto
    """Generate a model architeture, pointing to the given src_file """
    
    if not mean_file:
        mean_file = os.path.join(
            caffe_root,
            "examples",
            "c3d_feature_extraction",
            "sport1m_train16_128_mean.binaryproto")  ##TODO caffe root print later
    if not os.path.isfile(mean_file):
        print("[Error] mean cube file={} does not exist.".format(mean_file))
        sys.exit(-8)
   
    
    prototxt_content = '''
name: "DeepConv3DNet_Sport1M_Val"
layers {{
  name: "data"
  type: VIDEO_DATA
  top: "data"
  top: "label"
  image_data_param {{
    source: "{0}"
    use_image: true
    mean_file: "{1}"
    batch_size: 50
    crop_size: 112
    mirror: false
    show_data: 0
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: false
  }}
}}
# ----------- 1st layer group ---------------
layers {{
  name: "conv1a"
  type: CONVOLUTION3D
  bottom: "data"
  top: "conv1a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 64
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
layers {{
  name: "relu1a"
  type: RELU
  bottom: "conv1a"
  top: "conv1a"
}}
layers {{
  name: "pool1"
  type: POOLING3D
  bottom: "conv1a"
  top: "pool1"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 1
    stride: 2
    temporal_stride: 1
  }}
}}
# ------------- 2nd layer group --------------
layers {{
  name: "conv2a"
  type: CONVOLUTION3D
  bottom: "pool1"
  top: "conv2a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 128
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu2a"
  type: RELU
  bottom: "conv2a"
  top: "conv2a"
}}
layers {{
  name: "pool2"
  type: POOLING3D
  bottom: "conv2a"
  top: "pool2"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}
# ----------------- 3rd layer group --------------
layers {{
  name: "conv3a"
  type: CONVOLUTION3D
  bottom: "pool2"
  top: "conv3a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 256
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu3a"
  type: RELU
  bottom: "conv3a"
  top: "conv3a"
}}
layers {{
  name: "conv3b"
  type: CONVOLUTION3D
  bottom: "conv3a"
  top: "conv3b"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 256
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu3b"
  type: RELU
  bottom: "conv3b"
  top: "conv3b"
}}
layers {{
  name: "pool3"
  type: POOLING3D
  bottom: "conv3b"
  top: "pool3"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}

# --------- 4th layer group
layers {{
  name: "conv4a"
  type: CONVOLUTION3D
  bottom: "pool3"
  top: "conv4a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu4a"
  type: RELU
  bottom: "conv4a"
  top: "conv4a"
}}
layers {{
  name: "conv4b"
  type: CONVOLUTION3D
  bottom: "conv4a"
  top: "conv4b"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu4b"
  type: RELU
  bottom: "conv4b"
  top: "conv4b"
}}
layers {{
  name: "pool4"
  type: POOLING3D
  bottom: "conv4b"
  top: "pool4"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}

# --------------- 5th layer group --------
layers {{
  name: "conv5a"
  type: CONVOLUTION3D
  bottom: "pool4"
  top: "conv5a"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu5a"
  type: RELU
  bottom: "conv5a"
  top: "conv5a"
}}
layers {{
  name: "conv5b"
  type: CONVOLUTION3D
  bottom: "conv5a"
  top: "conv5b"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  convolution_param {{
    num_output: 512
    kernel_size: 3
    kernel_depth: 3
    pad: 1
    temporal_pad: 1
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu5b"
  type: RELU
  bottom: "conv5b"
  top: "conv5b"
}}

layers {{
  name: "pool5"
  type: POOLING3D
  bottom: "conv5b"
  top: "pool5"
  pooling_param {{
    pool: MAX
    kernel_size: 2
    kernel_depth: 2
    stride: 2
    temporal_stride: 2
  }}
}}
# ---------------- fc layers -------------
layers {{
  name: "fc6-1"
  type: INNER_PRODUCT
  bottom: "pool5"
  top: "fc6-1"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: 4096
    weight_filler {{
      type: "gaussian"
      std: 0.005
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu6"
  type: RELU
  bottom: "fc6-1"
  top: "fc6-1"
}}
layers {{
  name: "drop6"
  type: DROPOUT
  bottom: "fc6-1"
  top: "fc6-1"
  dropout_param {{
    dropout_ratio: 0.5
  }}
}}
layers {{
  name: "fc7-1"
  type: INNER_PRODUCT
  bottom: "fc6-1"
  top: "fc7-1"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: 4096
    weight_filler {{
    type: "gaussian"
      std: 0.005
    }}
    bias_filler {{
      type: "constant"
      value: 1
    }}
  }}
}}
layers {{
  name: "relu7"
  type: RELU
  bottom: "fc7-1"
  top: "fc7-1"
}}
layers {{
  name: "drop7"
  type: DROPOUT
  bottom: "fc7-1"
  top: "fc7-1"
  dropout_param {{
    dropout_ratio: 0.5
  }}
}}
layers {{
  name: "fc8-1"
  type: INNER_PRODUCT
  bottom: "fc7-1"
  top: "fc8-1"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: 487
    weight_filler {{
      type: "gaussian"
      std: 0.01
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
layers {{
  name: "prob"
  type: SOFTMAX
  bottom: "fc8-1"
  top: "prob"
}}
layers {{
  name: "accuracy"
  type: ACCURACY
  bottom: "prob"
  bottom: "label"
  top: "accuracy"
  #top: "prediction_truth"
}}'''.format(src_file, mean_file)

    with open(out_file, 'w') as f:
        f.write(prototxt_content)

    return

def create_src_and_output_file(video_file,start_frames, directory,frame_dir, feat_dir):
    video_id, video_ext = os.path.splitext(
        os.path.basename(video_file)
        ) # TODO check if this is correctx
 
    
    input_file = os.path.join(directory, 'input.txt')
    f_output_prefix_file = os.path.join(directory, 'output_prefix.txt')
    if os.path.isfile(input_file):
        f_input = open(input_file, 'a')
    else:
        f_input = open(input_file, 'w+')

    if os.path.isfile(f_output_prefix_file):
        f_output_prefix = open(f_output_prefix_file, 'a')
    else:
        f_output_prefix = open(f_output_prefix_file, 'w+')

    frame_dir = os.path.join(frame_dir,video_id)
    dummy_label = 0
    
    
    for start_frame in start_frames:
        
        f_input.write("{}/ {:d} {:d} \n".format(frame_dir,int(start_frame) + 1, int(dummy_label)))
        clip_id = os.path.join(
                feat_dir,
                video_id) + '/{0:06d}'.format(int(start_frame) + 1)
                
        f_output_prefix.write("{}\n".format(os.path.join(feat_dir, clip_id))) # ToDo : make extra output directory
    f_input.close()
    f_output_prefix.close()
    
    return f_output_prefix_file, input_file


def run_C3D_extraction(feature_prototxt, ofile, feature_layer, trained_model):
    """Extract C3D features by running caffee binary """

    almost_infinite_num = 9999999

    
    extract_bin = os.path.join(
            caffe_root,
            "build/tools/extract_image_features.bin"
            )

    if not os.path.isfile(extract_bin):
        print("[Error] Build facebook/C3D first, or make sure caffe_dir is "
              " correct")
        sys.exit(-9)

    feature_extraction_cmd = [
            extract_bin,
            feature_prototxt,
            trained_model,
            str(gpu_id),
            str(batch_size),
            str(almost_infinite_num),
            ofile,
            feature_layer,
            ]


    

    print ("[Info] Running C3D feature extraction: \"{}\"".format(
            ' '.join(feature_extraction_cmd)
            ))
    return_code = subprocess.call(feature_extraction_cmd)

    return return_code

def load_gazemaps(video_dir,gazemap_directory):
    ## loading groundtrouth gazemaps add if groundtruth here 
    video_path, video_id = os.path.split(video_dir)
    mat_file = h5py.File(os.path.join(gazemap_directory, video_id + '.mat'))
    key = list(mat_file.keys()) ## matfile namedtuple
    
    
    usernames = list(mat_file[key[0]]) ## username
    gazemap = mat_file[key[0]][usernames[0]]['gazemap49x49'][0]
    return gazemap 
    
    

def add_attention(video_dir, gazemap_directory, attention_mode = 'weighted'):
    #create new direcotry where the changed gazemaps are saved 
    images = sorted(glob.glob('{}/*.jpg'.format(video_dir)))
    frame_num = 0
    for image in images:
        frame_num += 1
        print('[INFO] Adding attention to frame num={}'.format(frame_num))
        img1 = cv2.imread(image)
        img_shape = img1.shape
        gazemap = load_gazemaps(video_dir, gazemap_directory)
        gazemap = np.resize(gazemap,img_shape)
        
        
        ## how to load gazemape?? need to save it 
        frame_with_att = np.multiply(img1, gazemap) ## gazemap has shape  framesx49x49
        frame_att_dir = os.path.join(video_dir, 'with_attention' + attention_mode)
        if not os.path.exists(frame_att_dir):
            os.mkdir(frame_att_dir)
        frame_att_file = os.path.join(frame_att_dir, '{0:06d}.jpg'.format(int(frame_num)))
        
        ### fuck also how to concatenate man this is all killing me
        cv2.imwrite(frame_att_file, frame_with_att)
    return

def process_c3d_features(feature_dir, c3d_layer):## read binary and save as np array 
    def read_binary(binary_feat):
        fin = open(binary_feat, 'rb')
        s = fin.read(20) #why 20 ?
        length = struct.unpack('i', s[4:8])[0]
        feature = fin.read(4*length)
        feature = struct.unpack("f"*length, feature)
        return list(feature)
    video_id = os.path.split(feature_dir)[1]

    print( "[Info] Collecting c3d features:{}".format(video_id))

    
    video_dir = feature_dir
    
    
    video_dir, video_name = os.path.split(video_dir)
    
    #feat_dir = feature_dir
    #if (not __force_computing__) and os.path.isfile(feature_dir+"/"+video_name+".c3d"):
     #   continue
    binary_features = sorted(glob.glob("{}/*/*.{}".format(video_dir,c3d_layer)))
    feats_movie = []
    
    for binary_feat in binary_features:
        tmp_feat = read_binary_blob(binary_feat)[1][1]
        feats_movie.append(tmp_feat)
    #if not feats_movie:
    #    print("[Warning] {} is too short; dummy c3d features are used".format(video_id))
        #temp = np.array(feats_movie, dtype=np.float32)
        #samples = np.round(np.linspace(0, len(temp) - 1, 64))
        #temp = temp[samples]
        #np.save(temp, feat_dir+"/"+video_name+"_c3d.npy")
    with open(video_dir  + '/'+ video_name + ".c3d", "wb") as f:
        
        pkl.dump(np.array(feats_movie,dtype = np.float32), f, protocol = 2) # so its compatible with python 


def main(video_file,video_directory, directory = None , input_output_file = None, trained_model = None,mean_file = None, use_attention = False, attention_mode = 'feature', feature_layer = 'conv5b'):
    

    ### creating new directory where pretrained model and binary file are saved
    
    model_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_files')
    if not os.path.exists(model_files_dir):
        os.mkdir(model_files_dir)
    new_path_trained =  os.path.join(model_files_dir,  "conv3d_deepnetA_sport1m_iter_1900000")
    new_path_binary =  os.path.join(model_files_dir,  "sport1m_train16_128_mean.binaryproto")

    if directory is None:
        directory  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c3d_prototxt')
        
        
        if not os.path.exists(directory):
            os.mkdir(directory)

    if trained_model is None and  not os.path.exists(new_path_trained ):
        #fined trained_model 
        trained_model = os.path.join(
        caffe_root,
        "examples",
        "c3d_feature_extraction",
        "conv3d_deepnetA_sport1m_iter_1900000"
            )
        #move trained model
        if not os.path.exists(new_path_trained ):
            os.rename(trained_model, new_path_trained)
        trained_model = new_path_trained

    else:
        if not os.path.exists(new_path_trained):
            os.rename(trained_model, new_path_trained)
        trained_model = new_path_trained
        
        
    if mean_file is None and not os.path.exists(new_path_binary):
        mean_file = os.path.join(
            caffe_root,
            "examples",
            "c3d_feature_extraction",
            "sport1m_train16_128_mean.binaryproto")

        if not os.path.isfile(mean_file):
            print( "[Error] mean cube file={} does not exist.".format(mean_file))
            sys.exit(-8)
        if not os.path.exists(new_path_binary):
            os.rename(mean_file, new_path_binary)
        mean_file = new_path_binary
    else:
        if not os.path.exists(new_path_binary):
            os.rename(mean_file, new_path_binary)
        mean_file = new_path_binary

    num_frames_per_clip = 16 # ~0.5 second
    force_overwrite = False
    feat_dir = '/data1/amelie/Hollywood2/vid_c3d2/'
    frame_dir = video_directory  ## == video directory 
    gazemap_dir = '/data1/amelie/Hollywood2/gazemap_cowork/'
    num_frames, fps = get_frame_count(video_file)
    video_id, video_ext = os.path.splitext(
        os.path.basename(video_file)
        )
   
    start_frames = [ i for i in range(0,num_frames,num_frames_per_clip)]
    
    extract_frames(video_file, start_frames, frame_dir, num_frames)
    
    ### after here I need to insert multiplying it with attention!!!

    if not os.path.exists(os.path.join(video_directory, video_id)):
        os.mkdir(os.path.join(video_directory, video_id))
    if not os.path.exists(os.path.join(feat_dir, video_id)):
            os.mkdir(os.path.join(feat_dir, video_id))

    if use_attention:
        
        
        add_attention(os.path.join(video_directory, video_id), gazemap_dir)
        video_directory = os.path.join(frame_dir, 'with_attention')
        feat_dir = os.path.join(feat_dir, 'with_attention')
        if not os.path.exists(video_directory):
            os.mkdir(video_directory)
        if not os.path.exists(feat_dir):
            os.mkdir(feat_dir)

    if input_output_file is None:
        input_file = os.path.join(directory,'input.txt')
        output_file = os.path.join(directory, 'output_prefix.txt')
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)
            
        output_file, input_file = create_src_and_output_file(video_file,start_frames, directory,frame_dir, feat_dir)
    else:
        input_file = input_output_file[0]
        output_file = input_output_file[1]


    #video_id = video_id.split('/')[-1]
    #frame_dir = os.path.join(video_directory, video_id)
    ### i think above directory split is a little wrong- needs to be fixed cause video_ ii is alreadt
    
    feature_prototxt = os.path.join(directory,'feature_extration.prototxt')
    generate_feature_prototxt(feature_prototxt, input_file, mean_file)

    #for start_frame in start_frames:

    
    if os.path.isfile(input_file) and os.path.getsize(input_file):
        import pdb; pdb.set_trace()
        
        return_code = run_C3D_extraction( ## also something going wrong here
            feature_prototxt,
            output_file,
            feature_layer,
            trained_model
            )

    ##video list get it with glob, but need to adjust above code to process multiple videos   
    

    
    process_c3d_features(os.path.join(feat_dir, video_id), feature_layer) #this here is wrong vid directoy ?? fram directory super confused 

    
if __name__ == "__main__":
    import argparse
    # To Do: add feat_dir, and frame_dir as argument 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vid_file', required = True, type = str)
    parser.add_argument('--vid_directory', required = True, type = str)
    parser.add_argument('--src_file', required = False, type = str)
    parser.add_argument('--output_prefix_file', required = False, type = str)
    parser.add_argument('--prefix_file_dir', required = False, type = str)
    parser.add_argument('--trained_model', required = False, type = str)
    parser.add_argument('--mean_file', required = False, type = str)
    parser.add_argument('--use_attention', required = False, type= bool)
    parser.add_argument('--attention_mode', required = False, type = str)
    parser.add_argument('--feature_layer', required = False, type = str)

    args = parser.parse_args()
    
    main(args.vid_file,args.vid_directory, use_attention = True)

