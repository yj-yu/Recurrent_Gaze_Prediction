from os import listdir
import os.path
from os.path import isdir, isfile, join
import sys
from PIL import Image
import numpy as np, h5py
from scipy import stats

from datetime import datetime
import cPickle as pkl
import hickle as hkl
from time import time
from scipy.sparse import coo_matrix, issparse

import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from util import log

from joblib import Parallel, delayed

CACHE_DIR = os.path.join('/data1', 'amelie', 'cache')
if not os.path.exists(CACHE_DIR): os.mkdir(CACHE_DIR)


def gather_filepaths(folder_path):
    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    for i in range(len(filenames)):
        filenames[i] = folder_path + filenames[i]
    return list(sorted(filenames))


def gather_foldernames(folder_path):
    foldernames = []
    for f in listdir(folder_path):
        if isdir(join(folder_path,f)) and  'action' in f:
            foldernames.append(f)

    return list(sorted(foldernames)) ## cause attention folder is added
    #return list(sorted([f for f in listdir(folder_path) if isdir(join(folder_path, f))]))

def apply_gaussian_filter(gazemaps, sigma):
    import scipy.ndimage

    assert len(gazemaps.shape) == 3
    for t in xrange(len(gazemaps)):
        g = scipy.ndimage.filters.gaussian_filter(gazemaps[t, :, :], sigma)
        g = g.astype(np.float32)
        if g.sum() == 0:
            continue
        g -= np.min(g)
        g /= np.max(g)
        gazemaps[t, :, :] = g
    return gazemaps

def wrap_object_array(*args):
    array = np.empty(len(args), dtype=np.object)
    for i in xrange(len(args)):
        array[i] = args[i]
    return array


# --------------------------------------------------------

class CRCDataSplits(object):
    def __init__(self):
        # each is of type CRCDataSet
        self.train = None
        self.valid = None
        self.test = None

    def __len__(self):
        return len(self.train) + len(self.valid) + len(self.test)

    def __repr__(self):
        s = '<CRCDataSplits object\n'
        if self.train: s += ' train : %d\n' % len(self.train)
        if self.valid: s += ' valid : %d\n' % len(self.valid)
        if self.test:  s += ' test  : %d\n' % len(self.test)
        s += '>'
        return s


class CRCDataSet(object):
    def __init__(self, images, gazemaps, fixationmaps, c3ds, pupils, clipnames, shuffle=False): # ???
        # wrap into numpy "object arrays" (rather than list)
        # so that non-contiguous index slicing is available
        self.images = np.asarray(images)
        self.c3ds = np.asarray(c3ds)
        self.pupils = np.asarray(pupils)
        self.gazemaps = np.asarray(gazemaps)
        self.clipnames = clipnames

        try:
            self.fixationmaps = np.asarray(fixationmaps)
        except:
            # XXX a dirty workaround.......orz......
            self.fixationmaps = wrap_object_array(*fixationmaps)
            
        assert len(self.images.shape) != 1
        assert len(self.gazemaps.shape) != 1
        assert len(self.gazemaps) == len(self.fixationmaps) == len(self.images) == len(self.c3ds)# == len(self.clipnames)
        self.epochs_completed = 0
        self.index_in_epoch = 0

        assert self.image_count() >= 0

        if shuffle:
            log.infov('Shuffling dataset...')
            batch_perm = list(range(self.image_count()))
            np.random.RandomState(3027300).shuffle(batch_perm)

            self.images = self.images[batch_perm, :]
            self.gazemaps = self.gazemaps[batch_perm, :]
            # XXX
            if len(self.fixationmaps.shape) > 1:
                self.fixationmaps = self.fixationmaps[batch_perm, :]
            else:
                self.fixationmaps = self.fixationmaps[batch_perm]
            self.c3ds = self.c3ds[batch_perm, :]
            self.pupils = self.pupils[batch_perm]
            log.infov('Shuffling done!!!')

    def __len__(self):
        return self.image_count()

    def __repr__(self):
        return 'CRC/Hollywood Dataset Split, %d instances' % len(self)

    def image_count(self):
        return len(self.c3ds) #.shape[0]

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.image_count():
            # Finished epochs
            self.epochs_completed += 1
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.image_count()
        end = self.index_in_epoch


        
        batch_indices = xrange(start,end)
        if len(self.pupils[batch_indices]) != len(self.clipnames[start:end]):
            import pdb; pdb.set_trace()
        
        return (self.images[batch_indices],
                self.gazemaps[batch_indices],
                self.fixationmaps[batch_indices],
                self.c3ds[batch_indices],
                self.pupils[batch_indices],
                self.clipnames[start:end]
                    )


def fill_gazemap(gazemap):
    # gazemap (360,7,7)
    gazelen = gazemap.shape[0]
    for i in range(gazelen):
        frm = gazemap[i,:,:].sum()
        if frm == 0:
            gazemap[i,:,:] = gazemap[i-1,:,:]
    return gazemap


def read_crc_data_set(frame_folder_path, gazemap_filename, c3d_filename, image_height,
                      image_width, gazemap_height, gazemap_width, dtype=np.float32,
                      fixation_original_scale=False,
                      msg=''):
    if msg:
        log.info(msg)


    frame_filepaths = gather_filepaths(frame_folder_path)
    clipnames = []
    clipnames2 = []

    images = []
    for filepath in frame_filepaths:
        clipname2 = filepath.split('/')[-2:]
            #clipname = clipname[0] +'/'+ clipname[1]
        clipnames2.append(clipname2[0])
    for filepath in frame_filepaths[15:len(frame_filepaths):5]:        
        clipname = filepath.split('/')[-2:]
        #clipname = clipname[0] +'/'+ clipname[1]
        clipnames.append(clipname[0])
        image = Image.open(filepath).convert('RGB')
       

        width, height = image.size
        if width != image_width or height != image_height:
            #print "Image resized!"
            image = image.resize((image_width, image_height), Image.ANTIALIAS)
        image = np.array(image)

        assert image.shape == (image_width, image_height, 3)
        images.append(image)
       
    images = np.stack(images, axis=0)
    assert len(images.shape) == 4 and images.shape[3] == 3 # RGB
    assert len(images) == len(clipnames)

    if dtype == tf.float32 or dtype == np.float32:
        # normalize pixel to [0, 1]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
  
    assert images.dtype == dtype


    assert images.shape == (len(images), image_width, image_height, 3)
    ##                 
    mat_file = h5py.File(gazemap_filename, 'r')
    gazemaps_list = [] 

    
    pupil_list = []
    for user_name in mat_file.values()[0].keys():
        # TODO: handle missing variables
        user_data_mat = mat_file.values()[0][user_name]

        if (gazemap_height, gazemap_width) == (7, 7):
            gazemap_keyname = 'gazemap7x7'
            gaussian_sigma = 0.3  # FIXME
        elif (gazemap_height, gazemap_width) == (14, 14):
            gazemap_keyname = 'gazemap7x7'
            gaussian_sigma = 0.6  # FIXME
        elif (gazemap_height, gazemap_width) == (49, 49): # doesn't exist in any of the userdata for data_set crc??
            gazemap_keyname = 'gazemap49x49'
            gaussian_sigma = 2.0  # FIXME
        elif (gazemap_height, gazemap_width) == (48, 48):  #doesn't exist in anye of the userdata for data_set crc??
            gazemap_keyname = 'gazemap48x48'
            gaussian_sigma = 2.0  # FIXME
        elif gazemap_height is None and gazemap_width is None:
            # Original scale.
            gazemap_keyname = 'gazemap'
            gaussian_sigma = 19
        else: raise ValueError("Unsupported gazemap shape")

        if gazemap_keyname not in user_data_mat.keys():
            print 'gazemap not exists (%s) : %s' % (user_name, user_data_mat.keys())
            continue
            #return None

        gazemaps = np.array(user_data_mat[gazemap_keyname], copy=False)

        if np.isnan(np.min(user_data_mat["pupilsize"])):
            continue
        pupil_list.append(np.squeeze(user_data_mat["pupilsize"]))
        gazemaps_list.append(gazemaps)
    #print(len(frame_filepaths), [len(gazemap) for gazemap in gazemaps_list])
        
    
    assert len(gazemaps_list) > 0  #if gazemap_list = 0 then no such length map in crc

    

    gazelen  = np.maximum(len(gazemaps_list[0]), len(gazemaps_list[1])) - 10 

    pupil_list = [pupil[15:gazelen:5] for pupil in pupil_list if (pupil.shape[0] > gazelen -1) ]
    
   
    pupils = np.mean(np.array(pupil_list), axis=0)

    gazemaps_list = [gazemap[15:gazelen:5] for gazemap in gazemaps_list if (len(gazemap) > gazelen - 1)]
    assert len(gazemaps_list) > 0 #and fixationmaps = np.sum(np.array(gazemaps_list), axis=0)

    fixationmaps = np.sum(np.asarray(gazemaps_list), axis=0)

    # covert to dense matrix here



    if issparse(fixationmaps[0]):
        fixationmaps = np.asarray([t.toarray() for t in fixationmaps])

    fixationmaps = np.swapaxes(fixationmaps, 1, 2)  # (width, height) --> (height, width)?
    
    
    assert fixationmaps.sum() > 0

    # apply gaussian filter framewise (in-place update to gazemaps)
    gazemaps = fixationmaps.astype(np.float32) / len(gazemaps_list)  #np.mean(np.array(gazemaps_list), axis=0)
    #gazemaps = np.swapaxes(gazemaps, 1, 2)  # (width, height) --> (height, width)? ALREADY APPLIED
    apply_gaussian_filter(gazemaps, gaussian_sigma)


    if fixation_original_scale:
        # override fixationmaps
        
        
        fixationmaps_list = []
        for user_name in mat_file.values()[0].keys():
            user_data_mat = mat_file.values()[0][user_name]

            if not 'fixation_t' in user_data_mat:
                continue
            # load sparse matrix from fixation_{t,r,c}
            fixation_t = user_data_mat['fixation_t']
            fixation_r = user_data_mat['fixation_r']
            fixation_c = user_data_mat['fixation_c']
            
            T, original_height, original_width = user_data_mat['gazemap'].shape
            fixationmaps = [ coo_matrix((original_height, original_width), dtype=np.uint8) ] * T
            # construct from fixation point to fixation map (sparse binary matrix)
            for t, r, c in zip(fixation_t, fixation_r, fixation_c):
                fixationmaps[t] = coo_matrix( ([1], ([r], [c])),
                                             shape=(original_height, original_width),
                                             dtype=np.uint8 )
            fixationmaps_list.append(fixationmaps)

        # a huge duplicates .......
        
        
        fixationmaps_list = [gazemap[15:gazelen:5] for gazemap in fixationmaps_list if (len(gazemap) > gazelen - 1)]

        fixationmaps = np.sum(np.asarray(fixationmaps_list), axis=0)
        if issparse(fixationmaps[0]):
            fixationmaps = np.asarray([t.toarray() for t in fixationmaps])
        fixationmaps = np.swapaxes(fixationmaps, 1, 2)  # (width, height) --> (height, width)?

    assert ( len(fixationmaps) == len(gazemaps))
    c3d = pkl.load(open(c3d_filename,'rb'))
    
    # remove single dimensional entries
    c3d = np.squeeze(c3d)
    assert c3d.shape[-2:] == (7, 7)
    
    '''
    For some unknown reason, gaze data is in deficiency (shorther than images)
    which is a dirty workaround (the length should have been equal beforehand)
    we are running out of time.. -- ONLY THE CASE FOR HOLLYWOOD DATASET - lets investigate tomorrow 
    '''

    n_frames = min(len(images), len(gazemaps), len(fixationmaps), len(c3d), len(pupils), len(clipnames))
    print 'gazelen : ', gazelen, 'n_frames :', n_frames, ' old images/gazemaps length:', len(images), len(gazemaps)

    images = images[:n_frames]
    gazemaps = gazemaps[:n_frames]
    clipnames = clipnames[:n_frames]
    fixationmaps = fixationmaps[:n_frames]
    c3d = c3d[:n_frames]
    pupils = pupils[:n_frames]
    assert n_frames > 0
    assert len(images)== len(clipnames)
    assert images.shape[-1] == 3
    assert c3d.shape[-2:] == (7, 7)
    
    mat_file.close()
    
    return CRCDataSet(images, gazemaps, fixationmaps, c3d, pupils, clipnames, shuffle=False)



def read_crc_data_set_wrapper( (foldername, ctx),
                              image_height, image_width,
                              gazemap_height, gazemap_width,
                              dtype,
                              fixation_original_scale=False,
                              msg=''):

    DATA_VIDEO_FRAME = ctx['DATA_VIDEO_FRAME']
    DATA_GAZE_MAP = ctx['DATA_GAZE_MAP']
    DATA_C3D = ctx['DATA_C3D']
   
    crc_data_read = read_crc_data_set(
        DATA_VIDEO_FRAME + foldername + '/',        DATA_GAZE_MAP + foldername + '.mat',
        DATA_C3D + foldername + '.c3d',
        image_height, image_width,
        gazemap_height, gazemap_width,
        dtype=dtype,
        fixation_original_scale=fixation_original_scale,
        msg=msg
    )


    
    return crc_data_read

def seq2batch(data, seq_len):
    def chunks(l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]
    # For CRC, it's typically 360

    if type(data) == list: # this is probably wrong finding it out now
        data_len = len(data)
         
        
        
    else:
        data_len = data.shape[0]
        
    seqs = []
    if data_len > seq_len:
        num_parts = int(data_len / seq_len)
        eq_parts = data[:num_parts*seq_len]
        remainder = data[-seq_len:]
        # It should be equal length.
        eq_chunks = chunks(eq_parts, seq_len)
        seqs.extend(eq_chunks)
        seqs.append(remainder)
    else:
        
            
        # repeated to reach seq_len (only firt axis!!!!!!)
        tile_count = (seq_len/data_len + 1)
        if type(data) == list:
            repeated = np.tile(data,[tile_count])
            repeated = repeated[:seq_len]
            seqs.append(repeated)
        else:
            # tile along with onl fyirst axis. (e.g. (35,98,98,3)->(70,98,98,3))
            repeated = np.tile(data, [tile_count] + [1] * (len(data.shape)-1))
            repeated = repeated[:seq_len]
            seqs.append(repeated)
            # (35, ~, ~, ~) array
    return np.asarray(seqs)


def get_dataset_split_foldernames(dataset, with_attention):

    if dataset == 'crc':
        DATA_PATH = '/data1/amelie/CRC/'
        DATA_VIDEO_FRAME = DATA_PATH + 'vid_frm_96/'

        DATA_GAZE_MAP = DATA_PATH + 'gazemap_cowork.backup2/'

        DATA_C3D = DATA_PATH + 'vid_c3d/'

        log.infov("Loading CRC")
        foldernames = sorted(gather_foldernames(DATA_VIDEO_FRAME))

        print 'shuffling...'
        np.random.RandomState(0).shuffle(foldernames)

    elif dataset == 'hollywood2':
        DATA_PATH = '/data1/amelie/Hollywood2/'
        DATA_VIDEO_FRAME = DATA_PATH + 'vid_frm/'
        DATA_GAZE_MAP = DATA_PATH + 'gazemap_cowork/'
        if with_attention:
            attention =  'with_attention/'
        else:
            attention = ''
        DATA_C3D = DATA_PATH + 'vid_c3d2/' + attention 
        #DATA_C3D = DATA_PATH + 'vid_c3d/'

        log.infov("Loading Hollywood2")
        
        foldernames = list(sorted(gather_foldernames(DATA_VIDEO_FRAME)))

        foldernames.sort(key=lambda x: ('test' in x, x))  # train comes first, test comes later

    else:
        raise NotImplementedError(dataset)

    total_num = len(foldernames)

    # split instances.
    if dataset == 'crc':
        train_rate, val_rate = 0.6, 0.4
        train_offset = int(train_rate * total_num)
        val_offset = train_offset + int(val_rate * total_num)

    elif dataset == 'hollywood2':
        if total_num > 1600:   # full dataset
            log.info("Using official train/test split for H2")
            train_offset = 823             # XXX
            #val_offset = 823               # no validation?
            val_offset = 823 + (884-1)   #884-1      # XXX
        else:
            train_rate, val_rate = 0.5, 0.4
            train_offset = int(train_rate * total_num)
            val_offset = train_offset + int(val_rate * total_num)

    context = {
        'DATA_PATH' : DATA_PATH,
        'DATA_VIDEO_FRAME' : DATA_VIDEO_FRAME,
        'DATA_GAZE_MAP' : DATA_GAZE_MAP,
        'DATA_C3D' : DATA_C3D,
    }


    SEQ_LEN = 42         # omg hardcode.......


    split = {
        'train' : [(foldername, context) for foldername in foldernames[:train_offset]],
        'valid' : [(foldername, context) for foldername in foldernames[train_offset:val_offset]],
        'test'  : [(foldername, context) for foldername in foldernames[val_offset:]],
        'SEQ_LEN' : SEQ_LEN,
    }



    log.info('train size : %d', len(split['train']))
    log.info('valid size : %d', len(split['valid']))
    log.info('test  size : %d', len(split['test']))
    return split


def read_crc_data_sets(image_height, image_width,
                       gazemap_height, gazemap_width,
                       dtype=tf.int8, use_cache=True,
                       batch_norm = False,
                       max_folders=None,
                       split_modes=None,
                       dataset='crc',
                       with_attention = False,
                       fixation_original_scale=False,
                       parallel_jobs=8):
    
    if max_folders is not None:
        use_cache = False

    if dataset == 'crcxh2':
        split_crc = get_dataset_split_foldernames('crc', with_attention)
        split_h2  = get_dataset_split_foldernames('hollywood2', with_attention)
        split = {
            'train' : split_crc['train'] + split_h2['train'],
            'valid' : split_crc['valid'] + split_h2['valid'],
            'test'  : split_crc['test'] + split_h2['test'],
            'SEQ_LEN' : split_h2['SEQ_LEN'],
        }
        log.info('CRC+H2 train size : %d', len(split['train']))
        log.info('CRC+H2 valid size : %d', len(split['valid']))
        log.info('CRC+H2 test  size : %d', len(split['test']))
    else:
        split = get_dataset_split_foldernames(dataset, with_attention)

    SEQ_LEN = split['SEQ_LEN']

    # shuffle!
    rs = np.random.RandomState(0)
    log.info('Shuffling each of train/valid/test ...')
    rs.shuffle(split['train'])
    rs.shuffle(split['valid'])
    rs.shuffle(split['test'])

    if max_folders is not None:
        log.warn('Reducing due to max_folders ... %d', max_folders)
        split['train'] = split['train'][:max_folders]
        split['valid'] = split['valid'][:max_folders]
        split['test']  = split['test'][:max_folders]


    def read_data_lists(instances,is_parallel):
        images_list = []
        gazemaps_list = []
        fixationmaps_list = []
        c3d_list = []
        pupil_list = []
        clipnames = []
        data_set_results = []
        
        
            
        if is_parallel is True:
            log.warn('Using parallel pool of %d workers ...', parallel_jobs)
            with Parallel(n_jobs = parallel_jobs, verbose=10) as parallel:

                #run in parallel
                data_set_results = parallel(delayed(read_crc_data_set_wrapper)(
                                            (foldername, ctx),
                                            image_height, image_width,
                                            gazemap_height, gazemap_width,
                                            dtype=dtype,
                                            fixation_original_scale=fixation_original_scale,
                                            msg='[%d/%d] foldername: %s' % (i, len(instances), foldername)
                                            ) \
                    for i, (foldername, ctx) in enumerate(instances))
               
#error here when loading crcxh2????
               
                data_set_results = list(data_set_results)   # sync-barrier #seems unneccastu though ?

        else:  # allow for non-parallel to allow for debugging
           
            data_set_results = []
            for i, (foldername, ctx) in enumerate(instances):
                
                data_set_result = read_crc_data_set_wrapper(
                                            (foldername, ctx),
                                            image_height, image_width,
                                            gazemap_height, gazemap_width,
                                            dtype=dtype,
                                            fixation_original_scale=fixation_original_scale,
                                            msg='[%d/%d] foldername: %s' % (i, len(instances), foldername)

                                            )

                data_set_results.append(data_set_result)



                data_set_results = list(data_set_results)   # sync-barrier

        for data_set in data_set_results:

            if data_set is not None:
                clipnames.extend(seq2batch(data_set.clipnames, SEQ_LEN))
                images_list.extend(seq2batch(data_set.images, SEQ_LEN))
                gazemaps_list.extend(seq2batch(data_set.gazemaps, SEQ_LEN))
                fixationmaps_list.extend(seq2batch(data_set.fixationmaps, SEQ_LEN))
                pupil_list.extend(seq2batch(data_set.pupils, SEQ_LEN))
                c3d_list.extend(seq2batch(data_set.c3ds, SEQ_LEN))
          

        
        # Pupil size normalization. min - max
        zscore = stats.zscore(np.asarray(pupil_list))
        pupil_list = zscore.tolist()

        # Pupil size normalization. min - max
        maxx = np.asarray(pupil_list).max()
        minx = np.asarray(pupil_list).min()
        pupil_list = [(x - minx / (maxx - minx)) for x in pupil_list]
        assert len(images_list) == len(gazemaps_list) == len(fixationmaps_list)
        return images_list, gazemaps_list, fixationmaps_list, c3d_list, pupil_list, clipnames

    def _cached_evaluation(cache_file, fn, *args):

        _start_time = time()
        if use_cache and os.path.exists(cache_file):
            log.infov('Loading from cache %s ...' % cache_file)
            ret = hkl.load(cache_file)
        else:
            if not use_cache: print 'cache is disabled :('
            ret = fn(*args)

            if use_cache:
                log.infov('Persisting into cache %s ...' % cache_file)
                hkl.dump(ret, cache_file, mode='w')
        _end_time = time()
        log.info('Done, Elapsed time : %.3f sec' % (_end_time - _start_time))
        return ret
    
    if batch_norm == True:
        batch = "batched"
    else:
        batch = ""

    cache_file_splits = {

        split_mode: os.path.join(CACHE_DIR, 'datasets_{}_{}_{}_{}_{}_{}.{}.hkl'.format(
                                 dataset, image_height, image_width, gazemap_height, gazemap_width,batch, split_mode)

        ) for split_mode in ['train', 'valid', 'test']
    }

    # data split
    def _read_data_splits(split_mode):
        


        images_list, gazemaps_list, fixationmaps_list, c3d_list, pupil_list, clipnames = read_data_lists(split[split_mode], is_parallel = True) # set to False when debugging  


        log.warn(split_mode + ' length: %d', len(images_list))
        return images_list, gazemaps_list, fixationmaps_list, c3d_list, pupil_list, clipnames


    if isinstance(split_modes, (unicode, str)): split_modes = [split_modes]
    if split_modes is None: split_modes = ['train', 'valid', 'test'] # load all by default
     

    
    data = CRCDataSplits()
    if 'train' in split_modes:
        data.train = CRCDataSet(*_cached_evaluation(cache_file_splits['train'], _read_data_splits, 'train'))
    if 'valid' in split_modes:
        data.valid = CRCDataSet(*_cached_evaluation(cache_file_splits['valid'], _read_data_splits, 'valid'))
    if 'test' in split_modes:
         data.test =  CRCDataSet(*_cached_evaluation(cache_file_splits['test'], _read_data_splits, 'test'))
    
    return data




if __name__ == '__main__':
    import argparse
    global crc_data_sets

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--gazemap_size', type=int, default=49, choices=[7, 49, -1])
    parser.add_argument('--dataset', type=str, default='crc', choices=['crc', 'hollywood2', 'crcxh2'])
    parser.add_argument('--max_folders', type=int, default=None)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--fixation_original_scale', action='store_true')
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-j', '--parallel_jobs', type=int, default=8)
    args = parser.parse_args()

    if args.gazemap_size == -1: args.gazemap_size = None
    if args.parallel_jobs < 1: args.parallel_jobs=1

    # self-test
    #data_sets = read_crc_data_sets(96, 96, 7, 7)
    crc_data_sets = read_crc_data_sets(98, 98, args.gazemap_size, args.gazemap_size,
                                       dtype=np.float32,
                                       use_cache=args.cache,
                                       dataset=args.dataset,
                                       max_folders=args.max_folders,
                                       split_modes=['test'] if args.only_test else None,
                                       parallel_jobs=args.parallel_jobs,
                                       fixation_original_scale=args.fixation_original_scale,
                                       )

    batch_tuple = crc_data_sets.train.next_batch(5)
    
    print len(batch_tuple)
    print 'img', batch_tuple[0].shape
    print 'gaz', batch_tuple[1].shape
    print 'fix', batch_tuple[2].shape
    print 'c3d', batch_tuple[3].shape
    print 'pup', batch_tuple[4].shape
    print 'actionfold', len(batch_tuple[5])

    if args.interactive:
        from IPython import embed; embed()  # XXX DEBUG
