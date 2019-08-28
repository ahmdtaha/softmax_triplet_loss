import os
import time
import json
import numpy as np
import tensorflow as tf
from pydoc import locate
from utils import user_io
import constants as const
from utils import os_utils
from utils import log_utils
from ranking import center_loss
from ranking import triplet_semi
from ranking import triplet_hard
import utils.tb_utils as tb_utils
import utils.tf_utils as tf_utils
from nets.conv_embed import ConvEmbed
from data_sampling.quick_tuple_loader import QuickTupleLoader
from data_sampling.triplet_tuple_loader import TripletTupleLoader
from config.base_config import BaseConfig



def create_triplet_samples(label_train,cfg):
    ## Create Triplet Indexes in advance

    training_samples_required = cfg.train_iters * cfg.caffe_iter_size * cfg.batch_size;
    samples_path = os.path.join(cfg.db_path,'{}_samples.txt'.format(training_samples_required))
    if os.path.exists(samples_path):
        sampling_idxs = os_utils.txt_read(samples_path)
        if len(sampling_idxs) < training_samples_required:
            print('Something is wrong with the saved {}'.format(samples_path))
            quit()
        else:
            sampling_idxs = np.array(sampling_idxs ,dtype=np.int32)

    else:
        dataset_size = len(label_train)
        num_epochs = int(np.ceil(training_samples_required / dataset_size))
        sampling_idxs = np.ones(num_epochs * dataset_size,
                                dtype=np.int32) * -1  ## assert later that no -1 exists in the list
        # available_sample_mask = np.ones(dataset_size,dtype=bool)
        insertion_ptr = 0
        available_sample_mask = np.ones(dataset_size, dtype=bool)
        num_classes = len(np.unique(label_train))
        per_class_mask = np.zeros((num_classes, len(label_train)), dtype=np.bool)
        for cls in range(num_classes):
            per_class_mask[cls, :] = label_train == cls

        for i in range(num_epochs):
            available_sample_mask.fill(True)
            counter = dataset_size
            if i % 100 == 0:  # Track progress
                print(i, num_epochs)
            while counter > 0:
                sampled_class = np.random.choice(np.unique(label_train[available_sample_mask]))
                class_samples = np.where(np.logical_and(per_class_mask[sampled_class, :], available_sample_mask))[0]
                num_samples = min(cfg.Triplet_K, len(class_samples))
                selected_samples_idxs = np.random.choice(class_samples, num_samples, replace=False)

                sampling_idxs[insertion_ptr:insertion_ptr + num_samples] = selected_samples_idxs
                insertion_ptr = insertion_ptr + num_samples
                counter -= num_samples
                available_sample_mask[selected_samples_idxs] = False

        os_utils.txt_write(samples_path,sampling_idxs)

    return sampling_idxs

def touch_dir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)
    else:
        print(path)
        if not user_io.ask_yes_no_question('Model dir already exists, continue -- override?'):
            quit()

def main(argv):
    cfg = BaseConfig().parse(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    save_model_dir = cfg.checkpoint_dir
    model_basename = os.path.basename(save_model_dir)
    touch_dir(save_model_dir)

    args_file = os.path.join(cfg.checkpoint_dir,'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(cfg), f, ensure_ascii=False, indent=2, sort_keys=True)
    # os_utils.touch_dir(save_model_dir)

    log_file = os.path.join(cfg.checkpoint_dir, cfg.log_filename + '.txt')
    os_utils.touch_dir(cfg.checkpoint_dir)
    logger = log_utils.create_logger(log_file)

    img_generator_class = locate(cfg.db_tuple_loader)
    args = dict()
    args['db_path'] = cfg.db_path
    args['tuple_loader_queue_size'] = cfg.tuple_loader_queue_size
    args['preprocess_func'] = cfg.preprocess_func
    args['batch_size'] = cfg.batch_size
    args['shuffle'] = False
    args['csv_file'] = cfg.train_csv_file
    args['img_size'] = const.max_frame_size
    args['gen_hot_vector'] = True
    train_iter = img_generator_class(args)
    args['batch_size'] = cfg.batch_size
    args['csv_file'] = cfg.test_csv_file
    val_iter = img_generator_class(args)

    trn_images, trn_lbls = train_iter.imgs_and_lbls()
    val_imgs, val_lbls = val_iter.imgs_and_lbls()


    with tf.Graph().as_default():
        if cfg.train_mode == 'semi_hard' or cfg.train_mode == 'hard' or cfg.train_mode == 'cntr':
            train_dataset = TripletTupleLoader(trn_images, trn_lbls,cfg).dataset
        elif cfg.train_mode == 'vanilla':
            train_dataset = QuickTupleLoader(trn_images, trn_lbls,cfg,is_training=True, shuffle=True,repeat=True).dataset
        else:
            raise NotImplementedError('{} is not a valid train mode'.format(cfg.train_mode))

        val_dataset = QuickTupleLoader(val_imgs, val_lbls,cfg, is_training=False,repeat=False).dataset
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        images_ph, lbls_ph = iterator.get_next()


        network_class = locate(cfg.network_name)
        model = network_class(cfg,images_ph=images_ph, lbls_ph=lbls_ph)


        if cfg.train_mode == 'semi_hard':
            pre_logits = model.train_pre_logits
            _, w, h, channels = pre_logits.shape
            embed_dim = cfg.emb_dim
            embedding_net = ConvEmbed(emb_dim=embed_dim, n_input=channels, n_h=h, n_w=w)
            embedding = embedding_net.forward(pre_logits)
            embedding = tf.nn.l2_normalize(embedding, axis=-1, epsilon=1e-10)
            margin = cfg.margin
            gt_lbls = tf.argmax(model.gt_lbls, 1);
            metric_loss = triplet_semi.triplet_semihard_loss(gt_lbls, embedding, margin)
            logger.info('Triplet loss lambda {}, with margin {}'.format(cfg.triplet_loss_lambda,margin))
            total_loss = model.train_loss + cfg.triplet_loss_lambda * tf.reduce_mean(metric_loss)
        elif cfg.train_mode == 'hard':
            pre_logits = model.train_pre_logits
            _, w, h, channels = pre_logits.shape
            embed_dim = cfg.emb_dim
            embedding_net = ConvEmbed(emb_dim=embed_dim, n_input=channels, n_h=h, n_w=w)
            embedding = embedding_net.forward(pre_logits)
            embedding = tf.nn.l2_normalize(embedding, axis=-1, epsilon=1e-10)
            margin = cfg.margin

            logger.info('Triplet loss lambda {}, with margin {}'.format(cfg.triplet_loss_lambda, margin))
            gt_lbls = tf.argmax(model.gt_lbls, 1);
            metric_loss  = triplet_hard.batch_hard(gt_lbls, embedding, margin)
            total_loss = model.train_loss + cfg.triplet_loss_lambda * tf.reduce_mean(metric_loss)
        elif cfg.train_mode == 'cntr':

            pre_logits = model.train_pre_logits
            _, w, h, channels = pre_logits.shape
            embed_dim = cfg.emb_dim
            embedding_net = ConvEmbed(emb_dim=embed_dim, n_input=channels, n_h=h, n_w=w)
            embedding = embedding_net.forward(pre_logits)
            embedding = tf.nn.l2_normalize(embedding, axis=-1, epsilon=1e-10)
            CENTER_LOSS_LAMBDA = 0.003
            CENTER_LOSS_ALPHA = 0.5
            num_fg_classes = cfg.num_classes
            gt_lbls = tf.argmax(model.gt_lbls, 1);
            center_loss_order, centroids, centers_update_op, appear_times, diff = center_loss.get_center_loss(embedding, gt_lbls,
                                                                                              CENTER_LOSS_ALPHA,
                                                                                              num_fg_classes)
            # sample_centroid = tf.reshape(tf.gather(centroids, gt_lbls), [-1, config.emb_dim])
            # center_loss_order = center_loss.center_loss(sample_centroid , embedding)
            logger.info('Center loss lambda {}'.format(CENTER_LOSS_LAMBDA))
            total_loss = model.train_loss + CENTER_LOSS_LAMBDA * tf.reduce_mean(center_loss_order)

        elif cfg.train_mode == 'vanilla':
            total_loss = model.train_loss

        logger.info('Train Mode {}'.format(cfg.train_mode))
        # variables_to_train = model.var_2_train();
        # logger.info('variables_to_train  ' + str(variables_to_train))

        trainable_vars = tf.trainable_variables()
        if cfg.caffe_iter_size > 1:  ## Accumulated Gradient
            ## Creation of a list of variables with the same shape as the trainable ones
            # initialized with 0s
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if cfg.train_mode == const.Train_Mode.CNTR:
            update_ops.append(centers_update_op)

        # print(update_ops)

        with tf.control_dependencies(update_ops):

            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf_utils.poly_lr(global_step,cfg)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

            if cfg.caffe_iter_size > 1:  ## Accumulated Gradient
                # grads = tf.Print(grads,[grads],'Grad Print');
                grads = optimizer.compute_gradients(total_loss, trainable_vars)
                # Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
                accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]
                iter_size = cfg.caffe_iter_size
                # Define the training step (part with variable value update)
                train_op = optimizer.apply_gradients([(accum_vars[i] / iter_size, gv[1]) for i, gv in enumerate(grads)],
                                                     global_step=global_step)

            else:
                grads = optimizer.compute_gradients(total_loss)
                train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # logger.info('=========================================================')
        # for v in tf.trainable_variables():
            # print('trainable_variables: ' + str(v.name) + '\t' + str(v.shape))
            # logger.info('trainable_variables: ' + str(v.name) + '\t' + str(v.shape))

        sess = tf.InteractiveSession()
        training_iterator = train_dataset.make_one_shot_iterator()
        validation_iterator = val_dataset.make_initializable_iterator()
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        tb_path = save_model_dir
        logger.info(tb_path)
        start_iter = tb_utils.get_latest_iteration(tb_path)

        train_writer = tf.summary.FileWriter(tb_path, sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()  # saves variables learned during training


        ckpt_file = tf.train.latest_checkpoint(save_model_dir)
        logger.info('Model Path {}'.format(ckpt_file))
        load_model_msg = model.load_model(save_model_dir, ckpt_file, sess, saver, load_logits=False)
        logger.info(load_model_msg)

        ckpt_file = os.path.join(save_model_dir, cfg.checkpoint_filename)


        val_loss = tf.summary.scalar('Val_Loss', model.val_loss)
        val_acc_op = tf.summary.scalar('Batch_Val_Acc', model.val_accuracy)
        model_acc_op = tf.summary.scalar('Split_Val_Accuracy', model.val_accumulated_accuracy)

        best_model_step = 0
        best_acc = 0
        logger.info('Start Training from {}, till {}'.format(start_iter,cfg.train_iters))
        for step in range(start_iter + 1, cfg.train_iters + 1):

            start_time_train = time.time()

            for mini_batch in range(cfg.caffe_iter_size - 1):
                feed_dict = {handle: training_handle}
                model_loss_value, accuracy_value, _ = sess.run(
                        [model.train_loss, model.train_accuracy, accum_ops], feed_dict)


            feed_dict = {handle: training_handle}
            model_loss_value, accuracy_value, _= sess.run([model.train_loss, model.train_accuracy, train_op],
                                                               feed_dict)
            if cfg.caffe_iter_size > 1:  ## Accumulated Gradient
                sess.run(zero_ops)

            train_time = time.time() - start_time_train

            if (step == 1 or step % cfg.logging_threshold == 0):
                logger.info(
                    'i {0:04d} loss {1:4f} Acc {2:2f} Batch Time {3:3f}'.format(step,model_loss_value,
                                                                                     accuracy_value,
                                                                                     train_time))

                if (step % cfg.test_interval == 0):
                    run_metadata = tf.RunMetadata()
                    tf.local_variables_initializer().run()
                    sess.run(validation_iterator.initializer)

                    _val_acc_op = 0
                    while True:
                        try:
                            # words, lbl = val_iter.next()

                            # feed_dict = {model.input: words, model.gt_lbls: lbl, is_training_flag: False}
                            feed_dict = {handle: validation_handle}
                            val_loss_op, batch_accuracy, accuracy_op, _val_acc_op, _val_acc, c_cnf_mat,macro_acc = sess.run(
                                [val_loss, model.val_accuracy, model_acc_op, val_acc_op, model.val_accumulated_accuracy,
                                 model.val_confusion_mat,model.val_per_class_acc_acc], feed_dict)
                        except tf.errors.OutOfRangeError:
                            logger.info('Val Acc {0}, Macro Acc: {1}'.format(_val_acc,macro_acc))
                            break

                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(val_loss_op, step)
                    train_writer.add_summary(_val_acc_op, step)
                    train_writer.add_summary(accuracy_op, step)
                    train_writer.flush()

                    if (step % 100 == 0):
                        saver.save(sess, ckpt_file)
                        if best_acc < _val_acc:
                            saver.save(sess, ckpt_file + 'best')
                            best_acc = _val_acc
                            best_model_step = step

                        logger.info('Best Acc {0} at {1} == {2}'.format(best_acc, best_model_step, model_basename))

        logger.info('Triplet loss lambda {}'.format(cfg.triplet_loss_lambda))
        logger.info('Mode {}'.format(cfg.train_mode))
        logger.info('Loop complete')
        sess.close()


if __name__ == '__main__':
    arg_db_name = 'cars'
    arg_net = 'inc4'
    arg_ckpt = 'test_{}_{}'.format(arg_db_name,arg_net)
    args = [
        '--gpu', '0',
        '--checkpoint_dir',arg_ckpt,
        '--db_name',arg_db_name ,
        '--net',arg_net ,

    ]
    main(args)








