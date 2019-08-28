import os
import utils.os_utils as os_utils
import tensorflow as tf

def get_tb_files(base_path):
    if(os.path.exists(base_path)):
        prefix = 'event'
        files = [os.path.join(base_path,f) for f in os.listdir(base_path) if (f.startswith(prefix) and not f.startswith('.'))];
        return files;
    return [];
def get_latest_iteration(tb_path):
    tb_files = get_tb_files(tb_path)
    if  len(tb_files)>0:
        # print('Found folder and files')
        # latest_filepath = os_utils.get_latest_file(tb_path, extension='lx')
        latest_filepath = max(tb_files, key=os.path.getctime)
        print(latest_filepath)
        tb_iter = tf.train.summary_iterator(latest_filepath)
        try:
            for e in tb_iter:
                last_step = e.step;
            print('Continue on previous TB file ', tb_path, ' with starting step', last_step);
        except:

            last_step = 0;
            print('Error: Continue on previous TB file ', tb_path, ' with starting step', last_step);

    else:
        print('New TB file *********** ', tb_path);
        last_step = 0;

    return last_step
