#!/usr/bin/python

import caffe
import glob
import lmdb
import numpy as np
from PIL import Image
import os
import sys

# Variables
img_width = 500
img_height = 500


# Paths
# PNG images
color_dir = '/origImgs'
# PNG images
# Per-pixel labels are stored in a gray image
label_dir = './OIGroundTruth'
output_dir = './lmdb/'


inputs = glob.glob(color_dir + '/*.png')

color_lmdb_name = output_dir + '/color-lmdb'
if not os.path.isdir(color_lmdb_name):
    os.makedirs(color_lmdb_name)
color_in_db = lmdb.open(color_lmdb_name, map_size=int(1e12))

label_lmdb_name = output_dir + '/label-lmdb'
if not os.path.isdir(label_lmdb_name):
    os.makedirs(label_lmdb_name)
label_in_db = lmdb.open(label_lmdb_name, map_size=int(1e12))

num_images = 30;
color_mean_color = np.zeros((3))


with color_in_db.begin(write=True) as color_in_txn:
    with label_in_db.begin(write=True) as label_in_txn:

        for in_idx, in_ in enumerate(inputs):
            img_name = os.path.splitext( os.path.basename(in_))[0]
            color_filename = color_dir + img_name + '.png'
            label_filename = label_dir + img_name + '.png'
            print(str(in_idx + 1) + ' / ' + str(len(inputs)))

            # load image
            im = np.array(Image.open(color_filename)) # or load whatever ndarray you need
            assert im.dtype == np.uint8            
            # RGB to BGR
            im = im[:,:,::-1]
            # in Channel x Height x Width order (switch from H x W x C)
            im = im.transpose((2,0,1))

            # compute mean color image
            for i in range(3):
                color_mean_color[i] += im[i,:,:].mean()
            num_images += 1

            #color_im_dat = caffe.io.array_to_datum(im)
            color_im_dat = caffe.proto.caffe_pb2.Datum()
            color_im_dat.channels, color_im_dat.height, color_im_dat.width = im.shape
            assert color_im_dat.height == img_height
            assert color_im_dat.width == img_width
            color_im_dat.data = im.tostring()
            color_in_txn.put('{:0>12d}'.format(in_idx), color_im_dat.SerializeToString())

            im = np.array(Image.open(label_filename)) # or load whatever ndarray you need
            assert im.dtype == np.uint8
            label_im_dat = caffe.proto.caffe_pb2.Datum()
            label_im_dat.channels = 1
            label_im_dat.height, label_im_dat.width = im.shape
            assert label_im_dat.height == img_height
            assert label_im_dat.width == img_width
            label_im_dat.data = im.tostring()
            label_in_txn.put('{:0>12d}'.format(in_idx), label_im_dat.SerializeToString())

    label_in_db.close()
color_in_db.close()

color_mean_color /= num_images
np.savetxt(output_dir + '/{}.csv'.format('color-mean'), color_mean_color, delimiter=",", fmt='%.4f')
