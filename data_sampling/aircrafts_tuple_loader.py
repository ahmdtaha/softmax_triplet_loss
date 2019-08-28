import numpy as np
import tensorflow as tf
import constants as const
from utils import os_utils
from multiprocessing.dummy import Pool as ThreadPool
from data_sampling.tuple_loader import BaseTupleLoader

class AircraftsTupleLoader(BaseTupleLoader):
    def __init__(self,args):
        BaseTupleLoader.__init__(self, args)
        self.img_path = args['db_path'] + '/fgvc-aircraft-2013b/data/images/'

        lbls = self.data_df['label']
        lbl2idx = np.sort(np.unique(lbls))

        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(lbls.values)]

        self.num_classes = len(self.lbl2idx_dict.keys())
        self.tuple_loader_queue_size = args['tuple_loader_queue_size']
        self.pool = ThreadPool(self.tuple_loader_queue_size );





        self.data_idx = 0

        self.preprocess_func = args['preprocess_func']
        self.batch_size = args['batch_size']
        self.img_size = args['img_size']
        self.gen_hot_vector = args['gen_hot_vector']
        print(
        'Data size ', self.data_df.shape[0], 'Num lbls', len(self.lbl2idx_dict.keys()), 'Batch Size', self.batch_size)


    def load_path_batch(self,img_idxs):
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
                img_path = self.img_path + str(imgs.iloc[img_idx]['file_name'])
                all_args.extend([img_path])



            results = self.pool.map(self.load_img_async, all_args)

            for j in range(num_threads):
                img = results[j]
                img_idx = img_idxs[num_img_in_batch + j]
                if img is None:
                    print('word failed at ', (self.data_idx + j))

                    img_path = self.img_path + imgs.iloc[img_idx]['file_name']
                    img = self.load_img_sync(img_path);

                # print(imgs.iloc[img_idx]['file_name'],imgs.iloc[img_idx]['label'],self.data_idx+j,num_img_in_batch+j)
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
        sample_lbls = np.zeros(current_batch_size ,dtype=np.int32)
        sample_imgs = np.zeros((current_batch_size, const.max_frame_size, const.max_frame_size, 3),dtype=np.float32);


        while (self.data_idx < num_files and num_img_in_batch < self.batch_size):
            num_threads = min(self.tuple_loader_queue_size, num_files - self.data_idx,self.batch_size - num_img_in_batch);
            all_args = []
            for j in range(num_threads):
                img_idx = self.data_permutation[self.data_idx + j]
                img_path = self.img_path + str(imgs.iloc[img_idx]['file_name'])
                all_args.extend([img_path])


            results = self.pool.map(self.load_img_async, all_args)

            for j in range(num_threads):
                img = results[j]
                img_idx = self.data_permutation[self.data_idx + j]
                if img is None:
                    print('word failed at ', (num_img_in_batch + j))
                    img_path = self.img_path + imgs.iloc[img_idx]['file_name']
                    img = self.load_img_sync(img_path);

                sample_lbls[num_img_in_batch+j] = self.lbl2idx_dict[imgs.iloc[img_idx]['label']]
                sample_imgs[num_img_in_batch+j, :, :, :] = img;

            self.data_idx += num_threads
            num_img_in_batch += num_threads

        if self.gen_hot_vector:
            labels_hot_vector = os_utils.hot_one_vector(sample_lbls, self.num_classes);
        else:
            labels_hot_vector = sample_lbls
        return sample_imgs, labels_hot_vector

    def next(self):
        return self.supervised_next();



