import os
import fast_fgvr_semi_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    num_trials = 1
    arg_db_name = 'aircrafts'
    arg_net = 'resnet50'
    arg_train_mode = 'semi_hard'
    lr = '0.01'
    for idx in range(num_trials):
        args = [
            '--gpu', '0',
            '--db_name', arg_db_name,
            '--net', arg_net,
            '--train_mode', arg_train_mode,
            '--margin', '0.2',
            '--caffe_iter_size', '10',
            '--logging_threshold', '500',
            '--train_iters', '120000',
            '--learning_rate', lr,
            '--aug_style', 'img',
            '--frame_size','299',
            '--checkpoint_suffix', '_lm1_aug_img_fixed_299_' + str(idx)

        ]


        fast_fgvr_semi_train.main(args)