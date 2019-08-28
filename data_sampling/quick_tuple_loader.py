import numpy as np
import tensorflow as tf
from pydoc import locate
import constants as const
from nets import img_augment
from nets import batch_augment



class QuickTupleLoader:
    def dataset_from_files(self,train_imgs, train_lbls,is_training,cfg,repeat=True,shuffle=False):



        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            #image_string = tf.Print(image_string,[filename,label],'img name ')
            image_decoded = tf.image.decode_jpeg(image_string,channels=3)
            # print(image_decoded.dtype)
            #image_decoded = tf.Print(image_decoded, [tf.shape(image_decoded)], 'shape ::')
            # image_resized = tf.image.resize_images(image_decoded, [const.max_frame_size, const.max_frame_size])
            # image_decoded = tf.image.resize_images(image_decoded, [const.max_frame_size, const.max_frame_size])
            # image_decoded = tf.cast(image_decoded, tf.uint8)

            return image_decoded, tf.one_hot(label, cfg.num_classes,dtype=tf.int64)

        filenames = tf.constant(train_imgs)
        labels = tf.constant(train_lbls,dtype=tf.int32)


        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))


        if shuffle:
            print('Will shuffle dataset ***')
            dataset = dataset.shuffle(len(train_imgs))

        if repeat:
            # dataset = dataset.shuffle(len(train_imgs))
            dataset = dataset.repeat(None)  # Repeat forever. Funny way of stating it.
        else:
            dataset = dataset.repeat(1)  # No Repeat


        dataset = dataset.map(_parse_function,num_parallel_calls=cfg.tuple_loader_queue_size)

        # preprocess_mod = locate(config.preprocessing_module)
        # func_name = preprocess_mod.preprocess_for_train_simple
        # if not is_training:
        #     func_name = preprocess_mod.preprocess_for_eval_simple
        # dataset = dataset.map(lambda im, lbl: (func_name (im,const.frame_height,const.frame_width), lbl))


        batch_size = cfg.batch_size



        if is_training:
            if cfg.aug_style == 'batch':
                dataset = dataset.batch(batch_size)
                dataset = dataset.map(lambda im_batch, lbl_batch: (batch_augment.augment(im_batch,cfg.preprocess_func,
                                                                                    horizontal_flip=True,
                                                                                    vertical_flip=False,
                                                                                    rotate=0, crop_probability=0,
                                                                                    color_aug_probability=0
                                                                                    ), lbl_batch))
            elif cfg.aug_style == 'img':
                dataset = dataset.map(lambda im, lbl: (img_augment.preprocess_for_train(im,cfg.frame_size,cfg.frame_size,preprocess_func=cfg.preprocess_func), lbl))
                dataset = dataset.batch(batch_size)
        else:
            if cfg.aug_style == 'batch':
                dataset = dataset.batch(batch_size)
                dataset = dataset.map(lambda im_batch, lbl_batch: (batch_augment.center_crop(im_batch,cfg.preprocess_func),lbl_batch))
            elif cfg.aug_style == 'img':
                dataset = dataset.map(lambda im, lbl: (img_augment.preprocess_for_eval(im,cfg.frame_size,cfg.frame_size,preprocess_func=cfg.preprocess_func), lbl))
                dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(1)
        return dataset



    def __init__(self,imgs,lbls ,cfg,is_training,repeat=True,shuffle=False):
        self.dataset = self.dataset_from_files(imgs, lbls,is_training,cfg,shuffle=shuffle,repeat=repeat)

