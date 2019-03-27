from __future__ import division
import tensorflow as tf
import numpy as np
import os
# import scipy.misc
import PIL.Image as pil
from SfMLearner import SfMLearner

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")  ### 由4改为1
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  ### 设置使用的GPU 3/20

def decompose(mul):   # 分解mul，返回一个最接近平方根的因子 3/27
    factor = int(np.ceil(np.sqrt(mul)))
    flag = True
    while flag and factor <= mul:
        if not mul % factor:  # 恰好整除
            return factor
        factor += 1
    return factor

def main(_):
    with open('data/kitti/test_files_eigen_2.txt', 'r') as f:
        test_files = f.readlines()
        test_files = [FLAGS.dataset_dir + t[:-1] for t in test_files]
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    basename = os.path.basename(FLAGS.ckpt_file)
    output_file = FLAGS.output_dir + '/' + basename
    sfm = SfMLearner()
    sfm.setup_inference(img_height=FLAGS.img_height,
                        img_width=FLAGS.img_width,
                        batch_size=FLAGS.batch_size,
                        mode='depth')
    saver = tf.train.Saver([var for var in tf.model_variables()]) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, FLAGS.ckpt_file)
        pred_all = []
        for t in range(0, len(test_files), FLAGS.batch_size):
            if t % 100 == 0:
                print('processing %s: %d/%d' % (basename, t, len(test_files)))
            inputs = np.zeros(
                (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3), 
                dtype=np.uint8)
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                fh = open(test_files[idx], 'rb')   ### 由'r'改为'rb' 3/20
                raw_im = pil.open(fh)
                scaled_im = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)
                inputs[b] = np.array(scaled_im)
                # im = scipy.misc.imread(test_files[idx])
                # inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
            pred = sfm.inference(inputs, sess, mode='depth')
            for b in range(FLAGS.batch_size):
                idx = t + b
                if idx >= len(test_files):
                    break
                pred_all.append(pred['depth'][b,:,:,0])
        np.save(output_file, pred_all)
        
        ### 生成预测的深度图像 3/27
        pred_disp = 255./np.array(pred_all)
        _,height,width = pred_disp.shape
        col = decompose(len(pred_disp))
        row = len(pred_disp) // col
        target = pil.new('RGB', (width*col, height*row*2))
        for i in range(row):
            for j in range(col):
                f = open(test_files[i*row+j], 'rb')   ### 由'r'改为'rb' 3/20
                raw_im = pil.open(f)
                src = raw_im.resize((FLAGS.img_width, FLAGS.img_height), pil.ANTIALIAS)

                target.paste(src, (j*width, 2*i*height, (j+1)*width, (2*i+1)*height)) # (左，上，右，下)
                src = pil.fromarray(pred_disp[i*row+j,:,:]).convert('L')
                target.paste(src, (j*width, (2*i+1)*height, (j+1)*width, 2*(i+1)*height))
                
        target.save(FLAGS.output_dir+'/depth.png')
        

if __name__ == '__main__':
    tf.app.run()