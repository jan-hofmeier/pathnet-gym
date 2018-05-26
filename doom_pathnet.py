# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time
import sys

# from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import NUM_GPUS
import visualize

import pathnet
import argparse

import gym
import gym.utils
from gym import wrappers
#import gym_doom
#from gym_doom.wrappers import *

import multiprocessing

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # mute tensorflow

FLAGS=None;
log_dir=None;

FIXED_VARS_BACKUP = None;
FIXED_VARS_IDX_BACKUP = None;

def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1-rate) + log_hi * rate
    return math.exp(v)

def train():
    #initial learning rate
    initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                        INITIAL_ALPHA_HIGH,
                                        INITIAL_ALPHA_LOG_RATE)

    # parameter server and worker information
    ps_hosts = np.zeros(FLAGS.ps_hosts_num,dtype=object);
    worker_hosts = np.zeros(FLAGS.worker_hosts_num,dtype=object);
    port_num=FLAGS.st_port_num;
    for i in range(FLAGS.ps_hosts_num):
        ps_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
        port_num+=1;
    for i in range(FLAGS.worker_hosts_num):
        worker_hosts[i]=str(FLAGS.hostname)+":"+str(port_num);
        port_num+=1;
    ps_hosts=list(ps_hosts);
    worker_hosts=list(worker_hosts);
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)


    if FLAGS.job_name == "ps":
        server.join();
    elif FLAGS.job_name == "worker":
        # gpu_assignment = FLAGS.task_index % NUM_GPUS
        # print("Assigning worker #%d to GPU #%d" % (FLAGS.task_index, gpu_assignment))
        # device=tf.train.replica_device_setter(
        #             worker_device="/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu_assignment),
        #             cluster=cluster);

        device=tf.train.replica_device_setter(
              worker_device="/job:worker/task:%d" % FLAGS.task_index,
              cluster=cluster);



        learning_rate_input = tf.placeholder("float")





        tf.set_random_seed(1);
        #There are no global network

        #lock = multiprocessing.Lock()

        #wrapper = ToDiscrete('constant-7')
        #env = wrapper(gym.make('gym_doom/DoomBasic-v0'))
        #env.close()


        # prepare session
        with tf.device(device):
            # flag for task
            flag = tf.get_variable('flag',[],initializer=tf.constant_initializer(0),trainable=False);
            flag_ph=tf.placeholder(flag.dtype,shape=flag.get_shape());
            flag_ops=flag.assign(flag_ph);
            # global step
            global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False);
            global_step_ph=tf.placeholder(global_step.dtype,shape=global_step.get_shape());
            global_step_ops=global_step.assign(global_step + global_step_ph)
            update_global_step = lambda sess, s: sess.run(global_step_ops, {global_step_ph: s})

            # score_set for genetic algorithm
            score_set=np.zeros(FLAGS.worker_hosts_num,dtype=object);
            score_set_ph=np.zeros(FLAGS.worker_hosts_num,dtype=object);
            score_set_ops=np.zeros(FLAGS.worker_hosts_num,dtype=object);
            for i in range(FLAGS.worker_hosts_num):
                score_set[i] = tf.get_variable('score'+str(i),[],initializer=tf.constant_initializer(-1000),trainable=False);
                tf.summary.scalar("scoreset/" + str(i),score_set[i])
                score_set_ph[i]=tf.placeholder(score_set[i].dtype,shape=score_set[i].get_shape());
                score_set_ops[i]=score_set[i].assign(score_set_ph[i]);
            update_score_set= lambda sess, s: sess.run(score_set_ops[FLAGS.task_index], { score_set_ph[FLAGS.task_index]: s})
            # fixed path of earlier task
            fixed_path_tf=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
            fixed_path_ph=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
            fixed_path_ops=np.zeros((FLAGS.L,FLAGS.M),dtype=object);
            for i in range(FLAGS.L):
                for j in range(FLAGS.M):
                    fixed_path_tf[i,j]=tf.get_variable('fixed_path'+str(i)+"-"+str(j),[],initializer=tf.constant_initializer(0),trainable=False);
                    fixed_path_ph[i,j]=tf.placeholder(fixed_path_tf[i,j].dtype,shape=fixed_path_tf[i,j].get_shape());
                    fixed_path_ops[i,j]=fixed_path_tf[i,j].assign(fixed_path_ph[i,j]);


            training_thread = A3CTrainingThread(FLAGS.task_index, initial_learning_rate, learning_rate_input,
                                                None,
                                                MAX_TIME_STEP, device=device, FLAGS=FLAGS)

            # parameters on PathNet
            vars=training_thread.pi.get_pathnet_vars()
            vars_init=np.zeros(len(vars),dtype=object)
            for i in range(len(vars)):
                vars_init[i] = tf.variables_initializer([vars[i]])
            # initialization
            init_op=tf.global_variables_initializer();
            # summary for tensorboard
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver();

            # Resume model if a model file is provided
            #if FLAGS.restore_dir:
            #    saver.restore(tf.get_default_session(), FLAGS.restore_dir)
            #    print("Loaded model from {}".format(FLAGS.restore_dir))

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == FLAGS.worker_hosts_num-1),
                                 global_step=global_step,
                                 logdir=FLAGS.log_dir,
                                 summary_op=summary_op,
                                 saver=saver,
                                 init_op=init_op)


        print("created Superviser")

        if (FLAGS.task_index == FLAGS.worker_hosts_num-1):
            print("chief")

        try:
            os.mkdir("./data/graphs")
        except:
            pass

        # config = tf.ConfigProto(
        #         device_count = {'GPU': 0}
        #     )
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.1

        with sv.managed_session(server.target) as sess, sess.as_default():
            lastTask = int(sess.run([flag])[0])
            recover = lastTask > 0
            lastTask = max(lastTask-1,0)
            if(FLAGS.task_index!=(FLAGS.worker_hosts_num-1)):
                 if recover: #give the other process time to restore geopaths
                     time.sleep(20)

                 for task in range(lastTask,2):
                    training_thread.set_training_stage(task)

                    while sess.run([flag])[0] < (task+1):
                        time.sleep(2)

                    # Set fixed_path
                    fixed_path=load_tf_fixed_path(sess, fixed_path_tf)
                    training_thread.set_fixed_path(fixed_path);
                    # set start_time
                    wall_t=0.0;
                    start_time = time.time() - wall_t
                    training_thread.set_start_time(start_time)
                    while True:
                        if sess.run([global_step])[0] > (MAX_TIME_STEP*(task+1)):
                            break
                        total_reward =training_thread.process(sess, update_global_step)
                        update_score_set(sess, total_reward)
            else:
                fixed_path=load_tf_fixed_path(sess, fixed_path_tf); #fixed_path=np.zeros((FLAGS.L,FLAGS.M),dtype=float)
                #vars_backup=np.zeros(len(vars_),dtype=object)
                #vars_backup=sess.run(vars_)
                winner_idx=0

                vis = visualize.GraphVisualize([FLAGS.M] * FLAGS.L, True)


                for task in range(lastTask,2):
                    # Generating randomly geopath
                    if not recover:
                        print("generate new geopath")
                        geopath_set=np.zeros(FLAGS.worker_hosts_num-1,dtype=object);
                        for i in range(FLAGS.worker_hosts_num-1):
                            geopath_set[i]=pathnet.get_geopath(FLAGS.L,FLAGS.M,FLAGS.N);
                            tmp=np.zeros((FLAGS.L,FLAGS.M),dtype=float);
                            for j in range(FLAGS.L):
                                for k in range(FLAGS.M):
                                    if((geopath_set[i][j,k]==1.0)or(fixed_path[j,k]==1.0)):
                                        tmp[j,k]=1.0;
                            pathnet.geopath_insert(sess,training_thread.pi.geopath_update_placeholders_set[i],training_thread.pi.geopath_update_ops_set[i],tmp,FLAGS.L,FLAGS.M);
                    else:
                        geopath_set=training_thread.pi.get_geopath_set(sess)
                        print("recovered geopaths")
                        recover = False
                    print("Geopath Setting Done");
                    sess.run(flag_ops,{flag_ph:(task+1)});
                    print("=============Task "+str(task+1)+"============");
                    score_subset=np.zeros(FLAGS.B,dtype=float);
                    score_set_print=np.zeros(FLAGS.worker_hosts_num,dtype=float);
                    rand_idx=np.arange(FLAGS.worker_hosts_num-1);
                    np.random.shuffle(rand_idx);
                    rand_idx=rand_idx[:FLAGS.B];
                    showPath(vis,geopath_set)
                    while sess.run([global_step])[0] <= (MAX_TIME_STEP*(task+1)):
                        # if (sess.run([global_step])[0]) % 1000 == 0:
                        #     print("Saving summary...")
                        #     tf.logging.info('Running Summary operation on the chief.')
                        #     summary_str = sess.run(summary_op)
                        #     sv.summary_computed(sess, summary_str)
                        #     tf.logging.info('Finished running Summary operation.')
                        #
                        #     # Determine the next time for running the summary.

                        flag_sum=0;
                        for i in range(FLAGS.worker_hosts_num-1):
                            score_set_print[i]=sess.run([score_set[i]])[0];
                        for i in range(len(rand_idx)):
                            score_subset[i]=sess.run([score_set[rand_idx[i]]])[0];
                            if(score_subset[i]==-1000):
                                flag_sum=1;
                                break;
                        if(flag_sum==0):
                            winner_idx=rand_idx[np.argmax(score_subset)];
                            print(str(sess.run([global_step])[0])+" Step Score: "+str(sess.run([score_set[winner_idx]])[0]));
                            for i in rand_idx:
                                if(i!=winner_idx):
                                    geopath_set[i]=np.copy(geopath_set[winner_idx]);
                                    geopath_set[i]=pathnet.mutation(geopath_set[i],FLAGS.L,FLAGS.M,FLAGS.N);
                                    tmp=np.zeros((FLAGS.L,FLAGS.M),dtype=float);
                                    for j in range(FLAGS.L):
                                        for k in range(FLAGS.M):
                                            if((geopath_set[i][j,k]==1.0)or(fixed_path[j,k]==1.0)):
                                                tmp[j,k]=1.0;
                                    pathnet.geopath_insert(sess,training_thread.pi.geopath_update_placeholders_set[i],training_thread.pi.geopath_update_ops_set[i],tmp,FLAGS.L,FLAGS.M);
                                sess.run(score_set_ops[i],{score_set_ph[i]:-1000})
                            rand_idx=np.arange(FLAGS.worker_hosts_num-1)
                            np.random.shuffle(rand_idx)
                            rand_idx=rand_idx[:FLAGS.B]
                            showPath(vis,geopath_set)
                        else:
                            time.sleep(2);
                    # fixed_path setting
                    fixed_path=geopath_set[winner_idx]

                    vis.set_fixed(decodePath(fixed_path), 'r' if task == 0 else 'g')
                    showPath(vis, geopath_set)
                    print('fix')
                    for i in range(FLAGS.L):
                        for j in range(FLAGS.M):
                            if(fixed_path[i,j]==1.0):
                                sess.run(fixed_path_ops[i,j],{fixed_path_ph[i,j]:1});
                    training_thread.set_fixed_path(fixed_path);

                    # backup fixed vars
                    # FIXED_VARS_BACKUP = training_thread.local_network.get_fixed_vars();
                    # FIXED_VARS_IDX_BACKUP = training_thread.local_network.get_fixed_vars_idx();

                    # initialization of parameters except fixed_path
                    vars_idx=training_thread.pi.get_unfixed_pathnet_vars_idx()

                    for i in range(len(vars_idx)):
                        if(vars_idx[i]==1.0):
                            sess.run(vars_init[i]);

                vis.waitForButtonPress()
        sv.stop();


def decodePath(p):
    return [np.where(l == 1.0)[0] for l in p]

def showPath(vis,geopath_set):
    vispaths = [np.array(decodePath(p)) for p in geopath_set]
    vis.show(vispaths, 'm')

def load_tf_fixed_path(sess, fixed_path_tf):
    fixed_path = np.zeros((FLAGS.L, FLAGS.M), dtype=float);
    for i in range(FLAGS.L):
        for j in range(FLAGS.M):
            if (sess.run([fixed_path_tf[i, j]])[0] == 1):
                fixed_path[i, j] = 1.0;
    return fixed_path


def main(_):
    FLAGS.ps_hosts_num+=1;
    FLAGS.worker_hosts_num+=1;
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
            "--ps_hosts_num",
            type=int,
            default=5,
            help="The Number of Parameter Servers"
    )
    parser.add_argument(
            "--worker_hosts_num",
            type=int,
            default=10,
            help="The Number of Workers"
    )
    parser.add_argument(
            "--hostname",
            type=str,
            default="localhost",
            help="The Hostname of the machine"
    )
    parser.add_argument(
            "--st_port_num",
            type=int,
            default=2222,
            help="The start port number of ps and worker servers"
    )
    parser.add_argument(
            "--job_name",
            type=str,
            default="",
            help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
            "--task_index",
            type=int,
            default=0,
            help="Index of task within the job"
    )

    parser.add_argument('--restore_dir', type=str, default=False,
                                            help='Restores Weights')

    parser.add_argument('--log_dir', type=str, default='./data/tensorboard/' + str(int(time.time())),
                                            help='Summaries log directry')
    parser.add_argument('--monitor_dir', type=str, default='/tmp/pathnet/atari/experiment-2',
                                            help='Gym Monitor log directry')
    parser.add_argument('--M', type=int, default=10,
                                            help='The Number of Modules per Layer')
    parser.add_argument('--L', type=int, default=4,
                                            help='The Number of Layers')
    parser.add_argument('--N', type=int, default=4,
                                            help='The Number of Selected Modules per Layer')
    parser.add_argument('--kernel_num', type=str, default='8,4,3',
                                            help='The Number of Kernels for each layer')
    parser.add_argument('--stride_size', type=str, default='4,2,1',
                                            help='Stride size for each layer')
    parser.add_argument('--B', type=int, default=3,
                                            help='The Number of Candidates for each competition')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
