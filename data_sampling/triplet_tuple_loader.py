import numpy as np
import tensorflow as tf
from nets import img_augment
from nets import batch_augment


class TripletTupleLoader:
    """
    This class provides a GPU-based fast triplet sampler.
    This code is taken from https://github.com/VisualComputingInstitute/triplet-reid/blob/master/train.py
    Given a list of filenames and their corrsponding labels, sample_k_fids_for_pid samples batch_k samples from a random selected class
    """
    def sample_k_fids_for_pid(self,pid, all_fids, all_pids, batch_k):
        """ Given a PID, select K FIDs of that specific PID. """
        possible_fids = tf.boolean_mask(all_fids, tf.equal(all_pids, pid))

        # The following simply uses a subset of K of the possible FIDs
        # if more than, or exactly K are available. Otherwise, we first
        # create a padded list of indices which contain a multiple of the
        # original FID count such that all of them will be sampled equally likely.
        count = tf.shape(possible_fids)[0]
        padded_count = tf.cast(tf.ceil(batch_k / tf.cast(count, tf.float32)), tf.int32) * count
        full_range = tf.mod(tf.range(padded_count), count)

        # Sampling is always performed by shuffling and taking the first k.
        shuffled = tf.random_shuffle(full_range)
        selected_fids = tf.gather(possible_fids, shuffled[:batch_k])

        return selected_fids, tf.fill([batch_k], pid)

    def dataset_from_files(self,train_imgs, train_lbls,cfg):
        train_imgs = np.array(train_imgs)
        train_lbls = np.array(train_lbls,dtype=np.int32)

        unique_pids = np.unique(train_lbls)
        class_per_batch = cfg.batch_size / cfg.Triplet_K
        if len(unique_pids) < class_per_batch:
            unique_pids = np.tile(unique_pids, int(np.ceil(class_per_batch / len(unique_pids))))

        dataset = tf.data.Dataset.from_tensor_slices(unique_pids)
        dataset = dataset.shuffle(len(unique_pids))
        num_classes = cfg.batch_size // cfg.Triplet_K
        dataset = dataset.take((len(unique_pids) // num_classes) * num_classes)
        dataset = dataset.repeat(None)  ## Such sampling is always used during training

        # For every PID, get K images.
        dataset = dataset.map(lambda pid: self.sample_k_fids_for_pid(
            pid, all_fids=train_imgs, all_pids=train_lbls, batch_k=cfg.Triplet_K))



        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            # image_string = tf.Print(image_string,[filename,label],'img name ')
            image_decoded = tf.image.decode_jpeg(image_string,channels=3)
            # print(image_decoded.dtype)
            # image_decoded = tf.Print(image_decoded, [tf.shape(image_decoded)], 'shape ')
            # image_resized = tf.image.resize_images(image_decoded, [const.max_frame_size, const.max_frame_size])
            # image_decoded = tf.image.resize_images(image_decoded, [const.max_frame_size, const.max_frame_size])
            # image_decoded = tf.cast(image_decoded, tf.uint8)

            return image_decoded, tf.one_hot(label, cfg.num_classes,dtype=tf.int64)

        dataset = dataset.apply(tf.contrib.data.unbatch())
        dataset = dataset.map(_parse_function,num_parallel_calls=cfg.tuple_loader_queue_size)

        batch_size = cfg.batch_size


        is_training = True ## Such sampling is always used during training
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
                dataset = dataset.map(lambda im, lbl: (
                img_augment.preprocess_for_train(im, cfg.frame_size, cfg.frame_size,
                                                 preprocess_func=cfg.preprocess_func), lbl))
                dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(1)
        return dataset



    def __init__(self,imgs,lbls,cfg):
        self.dataset = self.dataset_from_files(imgs, lbls,cfg)


