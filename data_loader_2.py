from __future__ import division
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None, 
                 batch_size=None, 
                 img_height=None, 
                 img_width=None, 
                 num_source=None, 
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'], 
            seed=seed, 
            shuffle=True)
        '''cam_paths_queue = tf.train.string_input_producer(   ### 注释掉3/10
            file_list['cam_file_list'], 
            seed=seed, 
            shuffle=True)'''
        self.steps_per_epoch = int(   ### 等于batch的个数3/8
            len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        '''cam_reader = tf.TextLineReader()    ### 注释掉3/10
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)   ### 读取器，一次只读取1行？3/13
        rec_def = []
        for i in range(9):   ### 指相机内参数矩阵K的9个元素3/8
            rec_def.append([1.])   ### rec_def=[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]] 3/13
        raw_cam_vec = tf.decode_csv(raw_cam_contents, 
                                record_defaults=rec_def)  ### 这个参数用于决定读取的数据类型，必须是lsit 3/13
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])'''

        # Form training batches
        '''src_image_stack, tgt_image, intrinsics = \
                tf.train.batch([src_image_stack, tgt_image, intrinsics], 
                               batch_size=self.batch_size)'''
        src_image_stack, tgt_image = \
                tf.train.batch([src_image_stack, tgt_image], 
                               batch_size=self.batch_size)     ### 删除intrinsics一项3/10

        '''# Data augmentation                              ### 删除数据增强3/13
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)      ### channel通道上级联？3/13
        #image_all, intrinsics = self.data_augmentation(
        #    image_all, intrinsics, self.img_height, self.img_width)
        #image_all = self.data_augmentation(      ### 删除intrinsics一项3/10   ### 删除数据增强3/13
        #    image_all, self.img_height, self.img_width)
        tgt_image = image_all[:, :, :, :3]      ### 第一帧为目标帧3/8
        src_image_stack = image_all[:, :, :, 3:]  ### 第二帧及以后的为源帧3/8
        #intrinsics = self.get_multi_scale_intrinsics(  ### 注释掉3/10
        #    intrinsics, self.num_scales)'''
        return tgt_image, src_image_stack  #, intrinsics

    '''def make_intrinsics_matrix(self, fx, fy, cx, cy):   ###输入相机内参数3/6
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]        ### 注释掉3/10
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics'''

    #def data_augmentation(self, im, intrinsics, out_h, out_w):     ### 删除intrinsics3/10
    def data_augmentation(self, im, out_h, out_w):             ###数据增强3/6
        # Random scaling
        #def random_scaling(im, intrinsics):
        def random_scaling(im):                        ### 删除intrinsics3/10
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)        ###随机放缩1-1.5倍3/6
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            '''fx = intrinsics[:,0,0] * x_scaling             ###对应修改相机参数3/6
            fy = intrinsics[:,1,1] * y_scaling              ### 注释掉3/10
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)'''
            return im   #, intrinsics

        # Random cropping
        #def random_cropping(im, intrinsics, out_h, out_w):      ###裁剪为指定尺寸3/6
        def random_cropping(im, out_h, out_w):              ###删掉intrinsics3/10
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            '''fx = intrinsics[:,0,0]                     ###对应修改相机参数3/6
            fy = intrinsics[:,1,1]                        ###注释掉3/10
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)'''
            return im   #, intrinsics
        #im, intrinsics = random_scaling(im, intrinsics)
        #im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = random_scaling(im)                         ###删掉intrinsics一项3/10
        im = random_cropping(im, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        return im   #, intrinsics

    def format_file_list(self, data_root, split):              ###数据集文件名放在data_root/train.txt文件中3/6
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        #cam_file_list = [os.path.join(data_root, subfolders[i],   ###注释掉3/10
        #    frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        #all_list['cam_file_list'] = cam_file_list             ###_cam.txt中存放相机参数信息3/6  注释掉3/10
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame   ### 假设中间的图像为目标图像3/8
        tgt_start_idx = int(img_width * (num_source//2))  ### 寻找目标图像的index3/8
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])  ### 这里为什么是img_width?3/8
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, 
                                   img_width, 
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    '''def get_multi_scale_intrinsics(self, intrinsics, num_scales):   ### 注释掉3/10
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale'''