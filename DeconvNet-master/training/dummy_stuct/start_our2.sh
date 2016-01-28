
CAFFE=/home/team76/new_caffe/caffe-master/build/tools/caffe
SOLVER=solver2_our.prototxt


$CAFFE train -solver $SOLVER

#send_notify_mail "039_voc_single_object_finetune_from_031 train script is finished"
