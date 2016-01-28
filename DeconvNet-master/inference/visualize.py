# Just enter the filename and extension, this code will automatically go fetch it in /home/team76/img
img_to_classify = 'bowser.jpg'
# Enter 1 to run just the neural network once on GPU and once on CPU
benchmark_iterations = 1


#imports
from benchmark_caffe import benchmark_caffe
import numpy as np
import time
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('/home/team76/reports/report'+time.strftime("%d-%m-%Y-%H-%M-%S")+'.pdf')
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    os.system("ipython %s/scripts/download_model_binary.py %s/models/bvlc_reference_caffenet" % (caffe_root, caffe_root))


caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) 
# mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_root + "../img/" + img_to_classify))
out = net.forward()
print("Predicted class is #{}.".format(out['prob'][0].argmax()))

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
plt.savefig(pp, format='pdf')
# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    os.system("ipython %s/data/ilsvrc12/get_ilsvrc_aux.sh" % caffe_root)
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]

benchmark_caffe(caffe, net, benchmark_iterations)
	
[(k, v.data.shape) for k, v in net.blobs.items()]

[(k, v[0].data.shape) for k, v in net.params.items()]

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
	data -= data.min()
	data /= data.max()

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

	# tile the filters into an image
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

	plt.imshow(data)
	plt.savefig(pp, format='pdf')

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat, padval=1)

filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))

feat = net.blobs['conv2'].data[0, :36]
vis_square(feat, padval=1)

feat = net.blobs['conv3'].data[0]
vis_square(feat, padval=0.5)

feat = net.blobs['conv4'].data[0]
vis_square(feat, padval=0.5)

feat = net.blobs['conv5'].data[0]
vis_square(feat, padval=0.5)

feat = net.blobs['pool5'].data[0]
vis_square(feat, padval=1)

feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig(pp, format='pdf')
 
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.savefig(pp, format='pdf')


feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.savefig(pp, format='pdf')

# load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    os.system("ipython %s/data/ilsvrc12/get_ilsvrc_aux.sh" % caffe_root)
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# sort top k predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print labels[top_k]
pp.close()
