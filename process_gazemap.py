"""
create gazemap 49x49 or 48x48 data
"""

from glob import glob
import os
import h5py
import numpy as np
import scipy.sparse

def resize_onehot_tensor(x, target_shape):
    """
    x is assumed to be 3-shape tensor [T x H x W]
    """
    assert len(x.shape) == 3
    assert len(target_shape) == 2

    T, W1, H1 = x.shape
    W2, H2 = target_shape

    print H1, W1, H2, W2
    ret = np.zeros((T, H2, W2), dtype=np.bool)

    for t, x, y in zip(*np.where(x > 0)):
        y_ = y * (H2 - 1.0) / (H1 - 1.0)
        x_ = x * (W2 - 1.0) / (W1 - 1.0)
        y_ = int(np.round(y_) + 1e-9)
        x_ = int(np.round(x_) + 1e-9)
        #print t, y, x, '=>', y_, x_
        ret[t, x_, y_] = 1 

    return ret


def fixation_points(x, target_shape):
    """
    x is assumed to be 3-shape tensor [T x H x W]
    """
    assert len(x.shape) == 3
    assert len(target_shape) == 2

    T, W1, H1 = x.shape
    W2, H2 = target_shape

    print H1, W1, H2, W2
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

    
def handle(mat_file, force=False):
    for user_name in mat_file.values()[0].keys():
        print '%s : %s ...' % (mat_file, user_name)
        user_data_mat = mat_file.values()[0][user_name]
        if not 'gazemap' in user_data_mat:
            print 'WTF !!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            continue

        gazemap_raw = user_data_mat['gazemap']
        
        if gazemap_raw[()].sum() == 0:
            print "Seoncd WTF!!!!"
            print "Deleting %s in %s" % (user_name, mat_file)
            del  mat_file.values()[0][user_name]
            continue
        
        if force and 'fixation' in user_data_mat:
            del user_data_mat['fixation']
        if 'fixation' not  in user_data_mat:
            fixation,_ ,_ ,_ = fixation_points(gazemap_raw.value, (49, 49))
            user_data_mat['fixation'] = fixation
        else: print '%s : fixation already done' % user_name

        
        if force and 'fixation48x48' in user_data_mat:
            del user_data_mat['fixation48x48']
        if 'fixation48x48' not in user_data_mat:
            fixation,_ ,_ ,_ = fixation_points(gazemap_raw.value, (48, 48))
            user_data_mat['fixation48x48'] = fixation
        else: print '%s : fixation 48x48 already done' % user_name

        
        if force and 'fixation49x49' in user_data_mat:
            del user_data_mat['fixation49x49']
        if 'fixation49x49' not in user_data_mat:
            fixation,_ ,_ ,_ = fixation_points(gazemap_raw.value, (49, 49))
            user_data_mat['fixation49x49'] = fixation
        else: print '%s : fxiation 49x49 already done' % user_name

            
        if force and  'gazemap49x49' in user_data_mat:
            del user_data_mat['gazemap49x49']
        if 'gazemap49x49' not in user_data_mat:
            gazemap_4949 = resize_onehot_tensor(gazemap_raw.value, (49, 49))
            user_data_mat['gazemap49x49'] = gazemap_4949
        else: print '%s : 49x49 already done' % user_name

        
        if force and 'gazemap48x48' in user_data_mat:
            del user_data_mat['gazemap48x48']
        if 'gazemap48x48' not in user_data_mat:
            gazemap_4848 = resize_onehot_tensor(gazemap_raw.value, (48, 48))
            user_data_mat['gazemap48x48'] = gazemap_4848
        else: print '%s : 48x48 already done' % user_name

        if force and 'fixation_t' in user_data_mat:
            del user_data_mat['fixation_t']
        if 'fixation_t' not in user_data_mat:
            _,fixation_t,_,_ = fixation_points(gazemap_raw.value, (49, 49))
            user_data_mat['fixation_t'] = fixation_t
        else: print '%s : fixation_t already done' % user_name

        if force and 'fixation_r' in user_data_mat:
            del user_data_mat['fixation_r']
        if 'fixation_r' not in user_data_mat:
            _,_,fixation_r,_ =fixation_points(gazemap_raw.value, (49, 49))
            user_data_mat['fixation_r'] = fixation_r
        else: print '%s : fixation_r already done' % user_name

        
        if force and 'fixation_c' in user_data_mat:
            del user_data_mat['fixation_c']
        if 'fixation_c' not in user_data_mat:
            _,_,_,fixation_c =fixation_points(gazemap_raw.value, (49, 49))
            user_data_mat['fixation_c'] = fixation_c
        else: print '%s : fixation_c already done' % user_name


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--override', default = False , type = bool)
    args = parser.parse_args()

    if 'AGENT_ID' in os.environ:
        AGENT_ID = int(os.environ['AGENT_ID'])
    else: AGENT_ID = None

    for i, f in enumerate(sorted(glob('*.mat'))):
        if AGENT_ID is not None:
            if i % 8 != AGENT_ID:
                continue

        print AGENT_ID, i, f
        mat_file = h5py.File(f, 'r+')
        handle(mat_file, force= args.override)
        mat_file.close()
