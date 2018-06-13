from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img, log_time_usage


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, tensorboard_dir, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):

    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod]

    style_features = {}
    img_height = img_width = 256  # change batch shape ffrom 256 to 128 to see if it is less memory intensive
    batch_shape = (batch_size,img_width,img_height,3)
    style_shape = (1,) + style_target.shape
    print("Style shape:", style_shape)
    print('Save tensorboard loags into: ', tensorboard_dir)

    # fix error : could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
    gpu_options = tf.GPUOptions(allow_growth=True,
    # per_process_gpu_memory_fraction=0.05
    )
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)

    print("Precompute style features")
    with tf.Graph().as_default(), tf.device('/cpu:0'), log_time_usage("Precompute style features done in"), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    print("Enter the main graph")
    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])

        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )
        tf.summary.scalar('content', content_loss)

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
        tf.summary.scalar('style_loss', style_loss)

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size
        tf.summary.scalar('tv_loss', tv_loss)
        loss = content_loss + style_loss + tv_loss
        tf.summary.scalar('loss', loss)

        # overall loss
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Merge all the summaries and write them out to data/logs (by default)
        merged = tf.summary.merge_all()

        if tensorboard_dir :
            print('Save TensorBoard summary in : ', tensorboard_dir)
            train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(tensorboard_dir + '/test', sess.graph)

        sess.run(tf.global_variables_initializer())
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            print("Start training of epoch:{} with n={} instances".format(epoch, num_examples))
            iterations = 0
            with log_time_usage("Epoch:{} done in".format(epoch)):
                start_time = step_time = time.time()
                nb_iterations = num_examples // batch_size
                while iterations * batch_size < num_examples:
                    batch_start_time = time.time()
                    curr = iterations * batch_size
                    step = curr + batch_size
                    X_batch = np.zeros(batch_shape, dtype=np.float32)
                    for j, img_p in enumerate(content_targets[curr:step]):
                       X_batch[j] = get_img(img_p, (img_width,img_height,3)).astype(np.float32)

                    iterations += 1
                    assert X_batch.shape[0] == batch_size

                    feed_dict = {
                       X_content:X_batch
                    }

                    if tensorboard_dir:
                        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
                        train_writer.add_summary(summary, iterations)
                    else:
                        sess.run([train_step], feed_dict=feed_dict)


                    batch_end_time = time.time()
                    batch_delta_time = batch_end_time - batch_start_time
                    if debug:
                        print("UID: %s, iterations: %s batch time: %s" % (uid, iterations, batch_delta_time))
                    is_print_iter = int(iterations) % print_iterations == 0
                    if slow:
                        is_print_iter = epoch % print_iterations == 0
                    is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                    should_print = is_print_iter or is_last
                    if should_print:
                        # compute avg batch time
                        end_time = time.time()
                        delta_time = end_time - step_time
                        total_time = end_time - start_time
                        step_time = end_time
                        avg_batch_time = delta_time / print_iterations
                        eta_in_hours = ((nb_iterations - iterations)*avg_batch_time
                            - nb_iterations*(epochs - 1 - epoch))/3600

                        to_get = [merged, style_loss, content_loss, tv_loss, loss, preds]
                        test_feed_dict = {
                           X_content:X_batch
                        }

                        tup = sess.run(to_get, feed_dict = test_feed_dict)
                        if tensorboard_dir:
                            test_writer.add_summary(summary, iterations)

                        summary, _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _loss)
                        if slow:
                           _preds = vgg.unprocess(_preds)
                        else:
                           saver = tf.train.Saver()
                           res = saver.save(sess, save_path)
                        time_info = (avg_batch_time, total_time, eta_in_hours)
                        yield(_preds, losses, iterations, epoch, time_info)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
