# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pathnet
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U

#from constants import ACTION_SIZEZ

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
    def __init__(self,
                 thread_index, # -1 for global
                 device="/cpu:0"):
        #self._action_size = ACTION_SIZEZ[self.training_stage]
        self._thread_index = thread_index
        self._device = device

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    # weight initialization based on muupan's code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_variable(self, weight_shape):
        input_channels    = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias     = tf.Variable(tf.random_uniform(bias_shape,     minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels    = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias     = tf.Variable(tf.random_uniform(bias_shape,     minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic PathNet Network
class GameACPathNetNetwork(GameACNetwork):
    def __init__(self,
                 name,
                 thread_index, # -1 for global
                 device="/cpu:0",
                 FLAGS="", geopath_set = None):
        GameACNetwork.__init__(self,
                               thread_index,
                               device)
        self.geopath_set = geopath_set
        self.task_index=FLAGS.task_index;
        scope_name = name +"net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:
            # First three Layers
            self.W_conv=np.zeros((FLAGS.L-1,FLAGS.M),dtype=object);
            self.b_conv=np.zeros((FLAGS.L-1,FLAGS.M),dtype=object);
            kernel_num=np.array(FLAGS.kernel_num.split(","),dtype=int);
            stride_size=np.array(FLAGS.stride_size.split(","),dtype=int);
            feature_num=[8,8,8];
            # last_lin_num=392;
            # last_lin_num=1280
            last_lin_num = 1408
            for i in range(FLAGS.L-1):
                for j in range(FLAGS.M):
                    if(i==0):
                        self.W_conv[i,j], self.b_conv[i,j] = self._conv_variable([kernel_num[i],kernel_num[i],4,feature_num[i]]);
                    else:
                        self.W_conv[i,j], self.b_conv[i,j] = self._conv_variable([kernel_num[i],kernel_num[i],feature_num[i-1],feature_num[i]]);

            # Last Layer in PathNet
            self.W_lin=np.zeros(FLAGS.M,dtype=object);
            self.b_lin=np.zeros(FLAGS.M,dtype=object);
            for i in range(FLAGS.M):
                self.W_lin[i], self.b_lin[i] = self._fc_variable([last_lin_num, 256])

            if self.geopath_set == None:
                # geopath_examples
                self.geopath_set=np.zeros(FLAGS.worker_hosts_num,dtype=object);
                for i in range(FLAGS.worker_hosts_num):
                    self.geopath_set[i]=pathnet.geopath_initializer(FLAGS.L,FLAGS.M);

                # geopathes placeholders and ops
                self.geopath_update_ops_set=np.zeros((FLAGS.worker_hosts_num,FLAGS.L,FLAGS.M),dtype=object);
                self.geopath_update_placeholders_set=np.zeros((FLAGS.worker_hosts_num,FLAGS.L,FLAGS.M),dtype=object);
                for s in range(FLAGS.worker_hosts_num):
                    for i in range(len(self.geopath_set[0])):
                        for j in range(len(self.geopath_set[0][0])):
                            tf.placeholder(self.geopath_set[s][i,j].dtype,shape=self.geopath_set[s][i,j].get_shape());
                            self.geopath_update_placeholders_set[s][i,j]=tf.placeholder(self.geopath_set[s][i,j].dtype,shape=self.geopath_set[s][i,j].get_shape());
                            self.geopath_update_ops_set[s][i,j]=self.geopath_set[s][i,j].assign(self.geopath_update_placeholders_set[s][i,j]);


            # state (input)
            self.s = tf.placeholder("float", [None, 160, 120, 4])

            for i in range(FLAGS.L):
                layer_modules_list=np.zeros(FLAGS.M,dtype=object);
                if(i==FLAGS.L-1):
                    net=tf.reshape(net,[-1,last_lin_num]);
                for j in range(FLAGS.M):
                    if(i==0):
                        layer_modules_list[j]=tf.nn.relu(self._conv2d(self.s,self.W_conv[i,j],stride_size[i])+self.b_conv[i,j])*self.geopath_set[self.task_index][i,j];
                    elif(i==FLAGS.L-1):
                        layer_modules_list[j]=tf.nn.relu(tf.matmul(net,self.W_lin[j])+self.b_lin[j])*self.geopath_set[self.task_index][i,j];
                    else:
                        layer_modules_list[j]=tf.nn.relu(self._conv2d(net,self.W_conv[i,j],stride_size[i])+self.b_conv[i,j])*self.geopath_set[self.task_index][i,j];
                net=np.sum(layer_modules_list);
            net=net/FLAGS.M;
            self.net=net
            # policy (output)
            # self.pi_source = tf.nn.softmax(tf.matmul(net, self.W_fc2_source) + self.b_fc2_source)
            # self.pi_target = tf.nn.softmax(tf.matmul(net, self.W_fc2_target) + self.b_fc2_target)
            self.pi = tf.nn.softmax(tf.matmul(net, self.W_fc2) + self.b_fc2)
            # value (output)
            v_ = tf.matmul(net, self.W_fc3) + self.b_fc3
            self.v = tf.reshape( v_, [-1] )

            # set_fixed_path
            self.fixed_path=np.zeros((FLAGS.L,FLAGS.M),dtype=float);

    def set_training_stage(self, ac_space):
        self.pdtype = make_pdtype(ac_space)

        # weight for policy output layer
        # self.W_fc2_source, self.b_fc2_source = self._fc_variable([256, ACTION_SIZEZ[0]])
        # self.W_fc2_target, self.b_fc2_target = self._fc_variable([256, ACTION_SIZEZ[1]])
        self.W_fc2, self.b_fc2 = self._fc_variable([256, self.pdtype.param_shape()[0]])

        # weight for value output layer
        self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

        logits = tf.matmul(self.net, self.W_fc2) + self.b_fc2
        self.pd = self.pdtype.pdfromflat(logits)
        self.vpred = tf.reshape(tf.matmul(self.net, self.W_fc3) + self.b_fc3, [-1])

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, self.s], [ac, self.vpred])


    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_geopath(self,sess):
        res=np.zeros((len(self.geopath_set[0]),len(self.geopath_set[0][0])),dtype=float);
        for i in range(len(res)):
            for j in range(len(res[0])):
                res[i,j]=self.geopath_set[self.task_index][i,j].eval(sess);
        return res;

    def set_fixed_path(self,fixed_path):
        self.fixed_path=fixed_path;

    def get_vars(self):
        res=[];
        for i in range(len(self.W_conv)):
            for j in range(len(self.W_conv[0])):
                if(self.fixed_path[i,j]==0.0):
                    res+=[self.W_conv[i,j]]+[self.b_conv[i,j]];
        for i in range(len(self.W_lin)):
            if(self.fixed_path[-1,i]==0.0):
                res+=[self.W_lin[i]]+[self.b_lin[i]];
        # if (self.training_stage == 0):
            # res+=[self.W_fc2_source]+[self.b_fc2_source];
            # res+=[self.W_fc3]+[self.b_fc3];
        res+=[self.W_fc2]+[self.b_fc2];
        res+=[self.W_fc3]+[self.b_fc3];
        return res;

    def get_trainable_variables(self):
        return self.get_vars()

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_vars_idx(self):
        res=[];
        for i in range(len(self.W_conv)):
            for j in range(len(self.W_conv[0])):
                if(self.fixed_path[i,j]==0.0):
                    res+=[1,1];
                else:
                    res+=[0,0];
        for i in range(len(self.W_lin)):
            if(self.fixed_path[-1,i]==0.0):
                res+=[1,1];
            else:
                res+=[0,0];
        # if (self.training_stage == 0):
        #     res+=[1,1,1,1];
        res += [1, 1, 1, 1];
        return res;