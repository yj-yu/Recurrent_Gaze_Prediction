import glob
from extract_C3D_features import main

frame_dir = '/data1/amelie/Hollywood2/vid_frm2/'
feat_dir = '/data1/amelie/Hollywood2/vid_c3d2/'
video_dir = '/data1/amelie/Hollywood2/AVIClips/'



video_list = glob.glob(video_dir +'actioncliptrain*.avi')

for video in video_list:
    main(video, frame_dir)
    main(video, frame_dir, use_attention = True)
    
    
video_list = glob.glob(video_dir +'actioncliptest*.avi')

for video in video_list:
    main(video, frame_dir)
    main(video, frame_dir, use_attention = True)
    
    
