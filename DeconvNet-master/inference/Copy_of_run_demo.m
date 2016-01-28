clear all; close all; clc;

%% startup
startup;
config.imageset = 'test';
config.cmap= './voc_gt_cmap.mat';
config.gpuNum = 0;
config.Path.CNN.caffe_root = '/home/team76/new_caffe/caffe-master';
config.save_root = './results/my_test/'; 
%% cache FCN-8s results
config.write_file = 1;
config.Path.CNN.script_path = '../training/dummy_stuct/100_100_relu';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/test_iter_30000.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/our_deploy.prototxt'];
config.im_sz = 792;

cache_FCN8s_results(config);

    %% generate EDeconvNet+CRF results

config.write_file = 1;
config.edgebox_cache_dir = './data/edgebox_cached/VOC2012_TEST';
config.Path.CNN.script_path = './DeconvNet';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DeconvNet_trainval_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DeconvNet_inference_deploy.prototxt'];
config.max_proposal_num = 50;
config.im_sz = 224;
config.fcn_score_dir = './results/FCN8s';

generate_EDeconvNet_CRF_results(config);