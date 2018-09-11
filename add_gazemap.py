import h5py
import tensorflow as tf
import numpy as np 
import glob

def resize_onehot_tensor(x, target_shape):
    """
    x is assumed to be 3-shape tensor [T x H x W]
    """
    assert len(x.shape) == 3
    assert len(target_shape) == 2

    T, W1, H1 = x.shape
    W2, H2 = target_shape

    #print H1, W1, H2, W2
    ret = np.zeros((T, W2, H2), dtype=np.bool) #Boolean ? WHY
    
    for t, x, y in zip(*np.where(x > 0)):
        y_ = y * (H2 - 1.0) / (H1 - 1.0)
        x_ = x * (W2 - 1.0) / (W1 - 1.0)
        y_ = int(np.round(y_) + 1e-9)
        x_ = int(np.round(x_) + 1e-9)
        
        #print t, y, x, '=>', y_, x_
        ret[t, x_, y_] = 1  #y, x notation is a little wrong here.?
    return ret


def fixation_points(x, target_shape):
    """
    x is assumed to be 3-shape tensor [T x H x W]
    """
    assert len(x.shape) == 3
    assert len(target_shape) == 2

    T, W1, H1 = x.shape
    W2, H2 = target_shape

    #print H1, W1, H2, W2
    ret = np.zeros((T, W2, H2), dtype=np.bool)
    rows = np.zeros((300))
    time, x_axis, y_axis = np.where(x>0)
    
    for t, x, y in zip(*np.where(x > 0)):
        y_ = y * (H2 - 1.0) / (H1 - 1.0)
        x_ = x * (W2 - 1.0) / (W1 - 1.0)
        y_ = int(np.round(y_) + 1e-9)
        x_ = int(np.round(x_) + 1e-9)
        #print t, y, x, '=>', y_, x_
        ret[t, x_, y_] = 1

    return ret, time, x_axis, y_axis



def add_missing_frame(eyegaze_full):
    "adds fixation if missing in frame"
    j = 1
    while eyegaze_full[0].sum() == 0:
        eyegaze_full[0] = eyegaze_full[j]
        j += 1
        if j == 360:
            assert eyegaze_full[0].sum() > 0, "all frames are 0, shouldn't get here,user should have been deleted "
            #filling in missing gazemap for framese
    for i in range(1,eyegaze_full.shape[0]):   
        j = i-1 
        while eyegaze_full[i].sum() == 0:
            eyegaze_full[i] = eyegaze_full[j]
            j -= 1
    assert eyegaze_full[i].sum() > 0 
    assert eyegaze_full.sum() == 360

    return eyegaze_full



def make_vignett(dataset_path = None):
    """
    Returns a list of labels that are contained in all runs
    NOT USED
    """
    if dataset_path is None:
        gazemap_path = "/data/common_datasets/CRC/gazemap_cowork.backup"
    else:
        gazemap_path = dataset_path
    run_datalist = sorted(glob.glob(gazemap_path + '/*.mat'))

    vignett = []

    for run in run_datalist:
        
        run_name = run.split('/')[-1].split('.')[0]
        f = h5py.File(run, 'r+', core = True )
        for label in f[run_name].keys():
            if label not in vignett:
                vignett += [label]
        f.close()

    return vignett


def add_gazemap(args, dataset_path = None):
    """ 
    Adding gazemap of required size to the VAS dataset. 
    --override [True|False]: deletes and recomputes the sizexsize gazemap
    --size [int]: size of the wanted gazemap

    """
    if dataset_path is None:
        gazemap_path = "/data/common_datasets/CRC/gazemap_cowork.backup"
    else:
        gazemap_path = dataset_path
    run_datalist = sorted(glob.glob(gazemap_path + '/*.mat'))

    vignett= make_vignett()
    
    size = args.size
    override = args.override 
    for run in run_datalist:
        
        run_name = run.split('/')[-1].split('.')[0] # run1_01.mat -> run1_01
        f = h5py.File(run, 'r+', core = True )
        print("adding gazemaps to %s " %(run_name))
        
        for   label in f[run_name].keys():

            if override is True:
                if 'gazemap%dx%d' %(size, size) in  f[run_name][label].keys():
                    del f[run_name][label]['gazemap%dx%d' % (size, size)] #del in case of rerun
            else:
                if 'gazemap%dx%d' %(size, size) in  f[run_name][label].keys():
                    print("gazemap%dx%d already exists, skipping..."  %(size, size))
                    continue 

                    
            if 'gazemap' not in f[run_name][label].keys():
                print("no gazemap, skipping %s" % (label))
                continue

            eyegaze_full = f[run_name][label]['gazemap'][()] # select whole array

            if eyegaze_full.sum() == 0:
                del f[run_name][label]['gazemap']
                print("Empty gazemaps! Deleting gazemap in %s for %s. " % (run_name, label))
                continue

            #filling in missing gazemap for frames
            eyegaze_full = add_missing_frame(eyegaze_full)
                
            gazemap = np.zeros((360,49,49))
            print("Adding %d x %d map to %s " % (size, size, label))
            gazemap = resize_onehot_tensor(eyegaze_full, (size, size))
            
            add_fixation(eyegaze_full,"fixation",f,size,run_name,label, override)
            add_fixation(eyegaze_full,"fixation_t",f,size,run_name, label, override)
            add_fixation(eyegaze_full,"fixation_r",f,size, run_name, label, override)
            add_fixation(eyegaze_full,"fixation_c",f,size, run_name, label,override)
            
            assert gazemap[15:360:5].sum() > 0
            assert gazemap[0].sum() == 1
            assert gazemap.sum() == 360
                
            f[run_name][label].__setitem__("gazemap%dx%d"% (size,size), gazemap)
                   
        print("finished %s" % (run_name))
                
        f.close()

def add_fixation(eyegaze_full,fix,f,size, run_name,label, override):
    """ 
    adding fixation, and fixation time, row, and column for reconstruction
    """                
    gazemap = np.zeros((360,size,size))
    if override and fix in f[run_name][label]:
        del f[run_name][label][fix]
    if fix == "fixation":
        fixations,_ ,_,_ = fixation_points(eyegaze_full, (size, size))
    elif fix == "fixation_t":
        _,fixations,_,_ = fixation_points(eyegaze_full, (size, size))
    elif fix == "fixation_r":
        _,_,fixations,_ = fixation_points(eyegaze_full, (size, size))
    elif fix == "fixation_c":
        _,_,_,fixations = fixation_points(eyegaze_full, (size, size))
                
                
    f[run_name][label].__setitem__(fix, fixations)
                   
    print("added   %s" % (fix))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', default = 49, type=int)
    parser.add_argument('--override', default = False , type = bool)
    args = parser.parse_args()
    add_gazemap(args)
    add_fixation(args)
