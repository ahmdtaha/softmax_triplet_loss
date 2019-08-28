import scipy.io as sio
# import configuration as config
import constants as const
import numpy as np
import imageio
import os
import cv2
from utils import os_utils
import time
import sys
import glob
import pandas as pd
# from data_sampling.preprocess_factory import PreProcessFactory
from multiprocessing.dummy import Pool as ThreadPool
import tensorflow as tf
from data_sampling.tuple_loader import BaseTupleLoader

class FLower102TupleLower(BaseTupleLoader):

    def __init__(self,args=None):
        BaseTupleLoader.__init__(self, args)
        self.img_path = args['db_path'] + '/jpg/'
        # self.img_processor = PreProcessFactory();

        lbls = self.data_df['label']
        lbl2idx = np.sort(np.unique(lbls))

        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(lbls.values)]

        self.num_classes = len(self.lbl2idx_dict.keys())
        self.tuple_loader_queue_size = args['tuple_loader_queue_size']
        self.pool = ThreadPool(self.tuple_loader_queue_size);

        # self.data_permutation = np.random.permutation(self.data_df.shape[0])

        self.data_idx = 0

        self.preprocess_func = args['preprocess_func']
        self.batch_size = args['batch_size']
        self.img_size = args['img_size']
        self.gen_hot_vector = args['gen_hot_vector']
        print(
            'Data size ', self.data_df.shape[0], 'Num lbls', len(self.lbl2idx_dict.keys()), 'Batch Size',
            self.batch_size)


    def load_img_batch(self,img_idxs):
        imgs = self.data_df
        current_batch_size = img_idxs.shape[0]
        sample_lbls = np.zeros(current_batch_size, dtype=np.int32)
        sample_imgs = np.zeros((current_batch_size, self.img_size, self.img_size, 3), dtype=np.float32);
        num_img_in_batch = 0
        while (num_img_in_batch < current_batch_size):
            num_threads = min(self.tuple_loader_queue_size,current_batch_size - num_img_in_batch);
            all_args = []
            for j in range(num_threads):
                img_idx = img_idxs[num_img_in_batch+j]
                img_path = self.img_path + imgs.iloc[img_idx]['file_name']
                all_args.extend([img_path])



            results = self.pool.map(self.load_img_async, all_args)

            for j in range(num_threads):
                img = results[j]
                img_idx = img_idxs[num_img_in_batch + j]
                if img is None:
                    print('word failed at ', (self.data_idx + j))

                    img_path = self.img_path + imgs.iloc[img_idx]['file_name']
                    img = self.load_img_sync(img_path);

                sample_lbls[num_img_in_batch+j] = self.lbl2idx_dict[imgs.iloc[img_idx]['label']]
                sample_imgs[num_img_in_batch+j, :, :, :] = img;

            num_img_in_batch += num_threads


        if self.gen_hot_vector:
            labels_hot_vector = os_utils.hot_one_vector(sample_lbls, self.num_classes);
        else:
            labels_hot_vector = sample_lbls

        return sample_imgs,labels_hot_vector

    def supervised_next(self):

        imgs = self.data_df

        num_files = len(self.data_permutation)
        if (self.data_idx >= num_files):
            self.data_permutation = np.random.permutation(imgs.shape[0])
            self.data_idx = 0
            raise tf.errors.OutOfRangeError(None, None, 'Epoch Reached, call again to reset')

        num_img_in_batch = 0
        current_batch_size = min(self.batch_size, num_files - self.data_idx)
        sample_lbls = np.zeros(current_batch_size, dtype=np.int32)
        sample_imgs = np.zeros((current_batch_size, const.max_frame_size, const.max_frame_size, 3), dtype=np.float32);

        while (self.data_idx < num_files and num_img_in_batch < self.batch_size):
            num_threads = min(self.tuple_loader_queue_size, num_files - self.data_idx,
                              self.batch_size - num_img_in_batch);
            all_args = []
            for j in range(num_threads):
                img_idx = self.data_permutation[self.data_idx + j]
                img_path = self.img_path + imgs.iloc[img_idx]['file_name']
                all_args.extend([img_path])

            results = self.pool.map(self.load_img_async, all_args)

            for j in range(num_threads):
                img = results[j]
                img_idx = self.data_permutation[self.data_idx + j]
                if img is None:
                    print('word failed at ', (num_img_in_batch + j))
                    img_path = self.img_path + imgs.iloc[img_idx]['file_name']
                    img = self.load_img_sync(img_path);

                sample_lbls[num_img_in_batch + j] = self.lbl2idx_dict[imgs.iloc[img_idx]['label']]
                sample_imgs[num_img_in_batch + j, :, :, :] = img

            self.data_idx += num_threads
            num_img_in_batch += num_threads

        if self.gen_hot_vector:
            labels_hot_vector = os_utils.hot_one_vector(sample_lbls, self.num_classes);
        else:
            labels_hot_vector = sample_lbls
        return sample_imgs, labels_hot_vector

    def next(self):
        return self.supervised_next()

    # def load(self,img_idxs):
    #     imgs = self.data_df
    #     current_batch_size = len(img_idxs)
    #     sample_lbls = np.zeros(current_batch_size, dtype=np.int32)
    #     sample_imgs = np.zeros((current_batch_size, const.max_frame_size, const.max_frame_size, 3), dtype=np.float32);
    #     img_idx_ptr = 0
    #     num_img_in_batch = 0
    #     while ():
    #         num_threads = min(self.tuple_loader_queue_size,current_batch_size - num_img_in_batch);
    #         all_args = []
    #         for j in range(num_threads):
    #             img_idx = self.data_permutation[img_idx_ptr + j]
    #             img_path = self.img_path + imgs.iloc[img_idx]['file_name']
    #             all_args.extend([img_path])
    #
    #         results = self.pool.map(self.load_img_async, all_args)
    #
    #         for j in range(num_threads):
    #             img = results[j]
    #             img_idx = self.data_permutation[img_idx_ptr + j]
    #             if img is None:
    #                 print('word failed at ', (num_img_in_batch + j))
    #                 img_path = self.img_path + imgs.iloc[img_idx]['file_name']
    #                 img = self.load_img_sync(img_path);
    #
    #             sample_lbls[num_img_in_batch + j] = self.lbl2idx_dict[imgs.iloc[img_idx]['label']]
    #             sample_imgs[num_img_in_batch + j, :, :, :] = img
    #
    #             img_idx_ptr += num_threads
    #         num_img_in_batch += num_threads
    #
    #     if self.gen_hot_vector:
    #         labels_hot_vector = os_utils.hot_one_vector(sample_lbls, self.num_classes);
    #     else:
    #         labels_hot_vector = sample_lbls
    #
    #     return sample_imgs, labels_hot_vector

#
#
#     def imgs2df(self,train_images,image_class_labels,image_paths,df):
#         for idx,img_id in enumerate(train_images):
#             ## Need to do -1 to avoid matlab 1-based indexing
#             img_lbl = image_class_labels[img_id-1]-1
#             img_path = image_paths[img_id-1]
#             df.loc[idx] = [img_lbl ,img_path ]
#         return df
#
#     def prepare_dataset(self):
#
#         image_class_labels = sio.loadmat(config.db_path + '/imagelabels.mat')['labels'].squeeze()
#         splits = sio.loadmat(config.db_path + '/setid.mat')
#
#         # image_paths = glob.glob(config.db_path+'/jpg/*.jpg')
#         # image_paths.sort()
#         image_paths = os_utils.get_files(config.db_path+'/jpg/', extension='.jpg', append_base=False)
#         image_paths.sort()
#          # ['trnid','valid','tstid']
#         train_images = splits['trnid'].squeeze()
#         val_images = splits['valid'].squeeze()
#         test_images = splits['tstid'].squeeze()
#
#         train_all_df = pd.DataFrame(columns=['label', 'file_name']);
#         val_all_df = pd.DataFrame(columns=['label', 'file_name']);
#         test_all_df = pd.DataFrame(columns=['label', 'file_name']);
#
#         train_all_df = self.imgs2df(train_images, image_class_labels, image_paths, train_all_df)
#         train_all_df.to_csv(config.db_path + '/lists/train_all_sub_list.csv')
#         print('All Train', train_all_df.shape[0])
#
#         val_all_df= self.imgs2df(val_images, image_class_labels, image_paths, val_all_df)
#         val_all_df.to_csv(config.db_path + '/lists/val_all_sub_list.csv')
#         print('All Val', val_all_df.shape[0])
#
#         test_all_df = self.imgs2df(test_images, image_class_labels, image_paths, test_all_df)
#         test_all_df.to_csv(config.db_path + '/lists/test_all_sub_list.csv')
#         print('All Test', test_all_df.shape[0])
#
#     def stats(self):
#         min_width = 100000
#         min_height = 100000
#         for i in range(8189):
#             img_path = self.img_path+'/image_%05d.jpg' %(i+1)
#             img = imageio.imread(img_path);
#             if(min_width> img.shape[1]):
#                 min_width = img.shape[1]
#
#             if (min_height > img.shape[0]):
#                 min_height = img.shape[0]
#         print('width ',min_width , ' height ',min_height )
#
# def vis_img(img,label,prefix,suffix):
#     imageio.imwrite(config.dump_path + prefix + '_' + str(label) + suffix + '.png',img)
#
# if __name__ == '__main__':
#     args = dict()
#     args['csv_file'] = config.train_csv_file
#     args['img_size'] = const.max_frame_size
#     args['gen_hot_vector'] = True
#     loader = FLower102TupleLower(args);
#     # loader.prepare_dataset()
#     # quit()
#     # flower_loader.preprocess();
#     # sys.exit(1)
#     start_time = time.time()
#     words, lbls,weight = loader.imgs_and_lbls(repeat=True);
#     print(len(words),type(words))
#     print(words[-5:])
#     print(lbls[-5:])
#     # print(np.argmax(lbls,axis=1))
#     elapsed_time = time.time() - start_time
#     print('elapsed_time :', elapsed_time)
#
#     sys.exit(1)


    # for batch_idx in range(words.shape[0]):
    #     lbl = lbls[batch_idx]
    #     if (np.prod(lbl.shape) > 1):
    #         lbl = np.argmax(lbls[batch_idx]);
    #     vis_img(words[batch_idx, :].astype(np.uint8), lbl, 'p_' + str(batch_idx), '_img')