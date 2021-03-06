from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader_2 import DataLoader
from nets_2 import *
from utils_2 import *

class SfMLearner(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,   ### 一组图像序列的帧数，论文是3帧？3/8
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            #tgt_image, src_image_stack, intrinsics = loader.load_train_batch()   ### 修改intrinsics为可训练参数3/10
            tgt_image, src_image_stack = loader.load_train_batch()   ### add 3/10
            #intrinsics = tf.Variable(
            #    initial_value=[[240.,0.,200.],[0.,240.,60.],[0.,0.,1.]],
            #    name='intrinsics')  ### add 3/13
            
            '''intrinsics_scale = tf.Variable(
                initial_value=tf.random_uniform([2],2,4,seed=9866),
                name='intrinsics_scale')  ### 放缩因子 3/14  改为随机初始化3/18
            intrinsics_trans = tf.Variable(
                initial_value=tf.random_uniform([2],5,30,seed=279),
                name='intrinsics_trans')   ### 平移因子 3/14  改为随机初始化3/18
            intrinsics = tf.concat(
                [60*tf.diag(intrinsics_scale),10*tf.expand_dims(intrinsics_trans,1)], 
                axis=1)  ### 拼接 3/14 乘以100 3/18  乘以1000 3/18
            intrinsics = tf.concat([intrinsics, tf.constant([[0.,0.,1.]])], axis=0)      ### 拼接最后一行 3/14'''
            ### 只训练1个相机参数，其余固定 3/21
            '''intrinsics_trans = tf.constant([[1.,0.,opt.img_width/2], 
                                            [0.,1.,opt.img_height/2], 
                                            [0.,0.,1.]]) ### 平移矩阵3/21
            scale = tf.Variable(
                initial_value=tf.random_uniform([1],10,20,seed=985),
                name='intrinsics_scale')  ### 放缩因子 3/21
            intrinsics_scale = tf.concat(
                [20* scale * tf.eye(num_rows=2,num_columns=3), tf.constant([[0.,0.,1.]])], 
                axis=0)  ### 拼接 3/21
            intrinsics = tf.matmul(intrinsics_trans, intrinsics_scale)  ### 3/21'''  ### delete 3/23
            ### 直接固定所有相机参数，放缩因子用一个错误值 3/23
            intrinsics = tf.constant([[100.,0.,opt.img_width/2],[0.,100.,opt.img_height/2],[0.,0.,1.]]) ### add 3/23
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(tgt_image, is_training=True)
            pred_depth = [1./d for d in pred_disp]

        with tf.name_scope("pose_and_explainability_prediction"):
            pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                pose_exp_net(tgt_image,
                             src_image_stack, 
                             do_exp=(opt.explain_reg_weight > 0),
                             is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            exp_loss = 0
            smooth_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []
            for s in range(opt.num_scales):
                if opt.explain_reg_weight > 0:
                    # Construct a reference explainability mask (i.e. all 
                    # pixels are explainable)
                    ref_exp_mask = self.get_reference_explain_mask(s)
                # Scale the source and target images for computing loss at the 
                # according scale.
                ### 对于低分辨率的图像也要计算loss，此时需要把源图像和目标图像都放缩3/8
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])                
                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight/(2**s) * \
                        self.compute_smooth_loss(pred_disp[s])

                for i in range(opt.num_source):
                    # Inverse warp the source image to the target image frame
                    curr_proj_image = projective_inverse_warp(   ### 用这个(utils.py)函数实现重建图像！！！3/8
                        curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                        tf.squeeze(pred_depth[s], axis=3),    ### 去掉所有维数为1的维度3/8
                        pred_poses[:,i,:],
                        intrinsics)
                        #intrinsics[:,s,:,:])      ### intrinsics [batch,3,3]已改为[3,3] 3/13
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    # Cross-entropy loss as regularization for the 
                    # explainability prediction
                    if opt.explain_reg_weight > 0:
                        curr_exp_logits = tf.slice(pred_exp_logits[s], 
                                                   [0, 0, 0, i*2], 
                                                   [-1, -1, -1, 2])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp_logits,
                                                      ref_exp_mask)
                        curr_exp = tf.nn.softmax(curr_exp_logits)
                    # Photo-consistency loss weighted by explainability
                    if opt.explain_reg_weight > 0:
                        pixel_loss += tf.reduce_mean(curr_proj_error * \
                            tf.expand_dims(curr_exp[:,:,:,1], -1))
                    else:
                        pixel_loss += tf.reduce_mean(curr_proj_error) 
                    # Prepare images for tensorboard summaries
                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                    else:
                        proj_image_stack = tf.concat([proj_image_stack, 
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack, 
                                                      curr_proj_error], axis=3)
                        if opt.explain_reg_weight > 0:
                            exp_mask_stack = tf.concat([exp_mask_stack, 
                                tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)
            #scaling_loss = tf.square(intrinsics_scale[0]-intrinsics_scale[1],
            #                        name='scaling_loss')  ### 约束xy方向的放缩因子尽量相同 3/20
            mean_depth = [tf.reduce_mean(pred_depth[i])-12. for i in range(len(pred_depth))]  ### add 3/21 delete 3/22 add 3/23
            ### 使pred_depth接近10.0,不能对4个scale求平均 3/21 delete 3/22
            depth_loss = 0.005*tf.reduce_sum(tf.square(mean_depth)) 
            total_loss = pixel_loss + smooth_loss + exp_loss #+ depth_loss
            ### add scaling 3/20 add depth 3/21 delete scaling 3/21 delete depth 3/22

        with tf.name_scope("train_op"):
            ### 刨除intrinsics参数 3/23
            train_vars = [var for var in tf.trainable_variables() if var.name!='data_loading/intrinsics_scale:0']
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            # self.grads_and_vars = optim.compute_gradients(total_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            #self.train_op = slim.learning.create_train_op(total_loss, optim) ### delete因为这里还是会训练所有参数 3/23
            self.train_op = optim.minimize(total_loss, var_list=train_vars)  ### 不训练intrinsics 3/23
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)
            
            '''train_intrinsics = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 'data_loading/intrinsics_scale:0') ### 单独训练 3/23
            boundaries_intrinsics = [15000]   ### 15k步之后停止对相机参数的训练 3/23
            learning_rate_intrinsics = [0.0002,0.]  ### 15k步之后学习率为0 3/23
            learning_rate_intrinsics = tf.train.piecewise_constant(
                self.global_step, 
                boundaries=boundaries_intrinsics, 
                values=learning_rate_intrinsics)  ### 分段常数衰减学习率 3/23
            optim_intrinsics = tf.train.AdamOptimizer(learning_rate_intrinsics, opt.beta1) ### add 3/23
            #self.train_op_intrinsics = slim.learning.create_train_op(total_loss, optim_intrinsics)  ### add 3/23
            self.train_op_intrinsics = optim_intrinsics.minimize(total_loss, var_list=train_intrinsics) ### add 3/23'''

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        #self.scaling_loss = scaling_loss   ### add 3/20
        self.depth_loss = depth_loss     ### 3/21 delete 3/22 add 3/23
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all
        #self.learning_rate_intrinsics = learning_rate_intrinsics ### 观察学习率 3/23 delete 3/23

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (opt.batch_size, 
                                int(opt.img_height/(2**downscaling)), 
                                int(opt.img_width/(2**downscaling)), 
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        tf.summary.scalar("depth_loss", self.depth_loss)  ### add 3/21 delete 3/22 add 3/23
        #tf.summary.scalar("learning_rate_intrinsics", self.learning_rate_intrinsics) ### add 3/23 delete 3/23
        for s in range(opt.num_scales):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1./self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i), 
                        tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i), 
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i), 
                    self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            #print(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS, 
            #                         scope='depth_prediciton')) ### 查看pre_depth变量名 3/21 打印为空[]
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step,
                    #"train_intrinsics": self.train_op_intrinsics   ### add 3/23 delete 3/23
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["depth"] = self.pred_depth   ### add 3/21 delete 3/22 add 3/23
                    #fetches["pixel_loss"] = self.pixel_loss  ### add 3/20
                    #fetches["exp_loss"] = self.exp_loss     ### add 3/20
                    #fetches["smooth_loss"] = self.smooth_loss ### add 3/20
                    #fetches["scaling_loss"] = self.scaling_loss ### add 3/20
                    #fetches["learning_rate_intrinsics"] = self.learning_rate_intrinsics ### 打印学习率 3/23 delete
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    #intrinsics = sess.graph.get_tensor_by_name('data_loading/intrinsics:0') ### add 3/13
                    #print(sess.run(intrinsics))   ### add 观察相机内参训练情况 3/13
                    #print()
                    #var1 = sess.graph.get_tensor_by_name('data_loading/intrinsics_scale:0') ### add 3/14
                    #var2 = sess.graph.get_tensor_by_name('data_loading/intrinsics_trans:0') ### add 3/14
                    #print(sess.run(var1))   ### add 观察相机内参训练情况 3/14
                    #print(sess.run(var2))   ### add 观察相机内参训练情况 3/14
                    #print("intrinsics learning rate: %f" % results["learning_rate_intrinsics"])### 观察学习率 3/23
                    #print("mean: %.4f" % np.mean(results["depth"][0]))  ### add 3/21
                    print("max: %.4f  min: %.4f  mean: %.4f" % \
                          (results["depth"][0].max(), \
                           results["depth"][0].min(), \
                           results["depth"][0].mean()))   ### 观察深度信息 3/21 add 3/23
                    #print("exp_loss    : %.4f" % sess.run(results["exp_loss"]))     ### 观察各项loss的量级 3/20
                    #print("pixel_loss   : %.4f" % sess.run(self.pixel_loss))   ### 观察各项loss的量级 3/20
                    #print("smooth_loss  : %.4f" % sess.run(self.smooth_loss))   ### 观察各项loss的量级 3/20
                    #print("scaling_loss : %.4f" % sess.run(self.scaling_loss))  ### 观察各项loss的量级 3/20
                    print()            ### add 3/14
                    #for var in tf.trainable_variables():           ### add 3/13
                    #    print(var.name)
                    #    if var.name == 'data_loading/intrinsics:0':  ### add 3/13
                    #        print(sess.run(var.value))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net(
                input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints
        ### 查看后发现训练好的原始网络预测的pred_depth均值为10左右
        ### 修改后的网络kitti_unknown学习到的均值只有1.2左右，可以看出来一定的深度信息
        ### intrinsics离谱时学习到的均值只有0.0999，且所有图像都得到相同的固定值 3/21

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
