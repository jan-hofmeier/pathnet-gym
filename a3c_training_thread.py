import pathnet
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys


from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

from game_state import GameState

# from game_ac_network import GameACFFNetwork, GameACLSTMNetwork, GameACPathNetNetwork
from game_ac_network import GameACPathNetNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM
from constants import USE_PATHNET
from constants import ROMZ
from constants import ACTION_SIZEZ

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 training_stage,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device,
                 FLAGS="",
                 task_index=""):

        clip_param = 0.2
        entcoeff = 0.01

        print("Initializing worker #{}".format(task_index))
        self.training_stage = training_stage
        self.thread_index = thread_index
        self.task_index = task_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.pi = GameACPathNetNetwork("pi", thread_index, device,FLAGS)
        self.oldpi = GameACPathNetNetwork("oldpi", thread_index, device, FLAGS, self.pi.geopath_set)

        self.local_t = 0


        # Setup losses and stuff
        # ----------------------------------------
        self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
        self.ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

        self.lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                                shape=[])  # learning rate multiplier, updated with schedule
        clip_param = clip_param * self.lrmult  # Annealed cliping parameter epislon

        self.ob = U.get_placeholder_cached(name="ob")

        self.vf_loss = tf.reduce_mean(tf.square(self.pi.vpred - self.ret))

        self.stage_dependend = []
        for i, _ in enumerate(ROMZ):
            ac = self.pi.pdtypes[i].sample_placeholder([None])
            kloldnew = self.oldpi.pds[i].kl(self.pi.pds[i])
            ent = self.pi.pds[i].entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            pol_entpen = (-entcoeff) * meanent

            ratio = tf.exp(self.pi.pds[i].logp(ac) - self.oldpi.pds[i].logp(ac))  # pnew / pold
            surr1 = ratio * self.atarg  # surrogate from conservative policy iteration
            surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * self.atarg  #
            pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

            total_loss = pol_surr + pol_entpen + self.vf_loss

            self.stage_dependend+=[(ac,meankl, meanent, pol_entpen,pol_surr,total_loss)]

        return


        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            self.local_network.get_vars(),
            self.gradients )



        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def set_training_stage(self, training_stage):
        self.training_stage = training_stage
        self.game_state = GameState(113 * self.task_index, ROMZ[training_stage], display=False,
                                     task_index=self.task_index)
        self.pi.set_training_stage(training_stage)
        self.oldpi.set_training_stage(training_stage)
        print("Setting training task to:  " + ROMZ[training_stage] + ", with action size: " + str(ACTION_SIZEZ[self.training_stage]))
        if training_stage == 1:
            self.game_state.close_env()

        (self.ac, self.meankl, self.meanent, self.pol_entpen, self.pol_surr, self.total_loss)=self.stage_dependend[training_stage]

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score,
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def processOld(self, sess, global_t, summary_writer, summary_op, score_input,score_ph,score_ops, geopath, FLAGS,score_set_ph,score_set_ops):

        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        start_local_t = self.local_t

        if FLAGS.use_lstm:
            start_lstm_state = self.local_network.lstm_state_out

        res_reward=-1000;
        # t_max times loop
        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)
            action = self.choose_action(pi_)

            states.append(self.game_state.s_t)
            actions.append(action)
            values.append(value_)

            # process game
            self.game_state.process(action)

            # receive game result
            reward = self.game_state.reward
            terminal = self.game_state.terminal

            self.episode_reward += reward

            # clip reward
            rewards.append( np.clip(reward, -1, 1) )

            self.local_t += 1

            # s_t1 -> s_t
            self.game_state.update()

            if terminal:
                terminal_end = True
                sess.run(score_ops,{score_ph:self.episode_reward});
                sess.run(score_set_ops,{score_set_ph:self.episode_reward});
                res_reward=self.episode_reward;
                self.episode_reward = 0
                self.game_state.reset()
                if USE_LSTM:
                    self.local_network.reset_state()
                break
        if(res_reward==-1000):
            res_reward=self.episode_reward;
        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            # a = np.zeros([ACTION_SIZEZ[self.training_stage]])
            a = np.zeros(max(ACTION_SIZEZ))
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)


        var_idx=self.local_network.get_vars_idx();
        gradients_list=[];
        for i in range(len(var_idx)):
            if(var_idx[i]==1.0):
                gradients_list+=[self.apply_gradients[i]];
        sess.run(gradients_list,
                feed_dict = {
                    self.local_network.s: batch_si,
                    self.local_network.a: batch_a,
                    self.local_network.td: batch_td,
                    self.local_network.r: batch_R,
                    self.learning_rate_input: cur_learning_rate} )



        if (self.task_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t,    elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t;


    def process(self, sess, global_t, summary_writer, summary_op, score_input,score_ph,score_ops, geopath, FLAGS,score_set_ph,score_set_ops):

        max_timesteps=int(LOCAL_T_MAX * 1.1)
        timesteps_per_actorbatch=256
        optim_epochs=4
        optim_stepsize=1e-3
        optim_batchsize=64
        gamma=0.99
        lam=0.95
        schedule='linear'
        adam_epsilon = 1e-5
        max_iters = 0
        max_episodes = 0
        max_seconds = 0

        start_local_t = self.local_t

        pi = self.pi
        oldpi = self.oldpi

        losses = [self.pol_surr, self.pol_entpen, self.vf_loss, self.meankl, self.meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = pi.get_trainable_variables()
        lossandgrad = U.function([self.ob, self.ac, self.atarg, self.ret, self.lrmult], losses + [U.flatgrad(self.total_loss, var_list)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in
                                                        zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([self.ob, self.ac, self.atarg, self.ret, self.lrmult], losses)

        U.initialize()
        adam.sync()

        # Prepare for rollouts
        # ----------------------------------------
        seg_gen = self.traj_segment_generator(pi, timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

        assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                    max_seconds > 0]) == 1, "Only one time constraint permitted"

        while True:
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************" % iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            assign_old_eq_new()  # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses, _, _ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_" + name, lossval)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()

        diff_local_t = self.local_t - start_local_t
        return diff_local_t;


    def set_fixed_path(self, fp):
        self.pi.set_fixed_path(fp)
        self.oldpi.set_fixed_path(fp)

    def traj_segment_generator(self, pi, horizon, stochastic):
        t = 0
        ac = self.game_state.get_ac_space().sample()  # not used, just so we have the datatype
        new = True  # marks if we're on first timestep of an episode
        ob = self.game_state.reset()

        cur_ep_ret = 0  # return in current episode
        cur_ep_len = 0  # len of current episode
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()

        while True:
            prevac = ac
            ac, vpred = pi.act(stochastic, ob)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            i = t % horizon
            if t > 0 and i == 0:
                yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                       "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                       "ep_rets": ep_rets, "ep_lens": ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []

            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, _ = self.game_state.step(ac)
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = self.game_state.reset()
            t += 1
            self.local_t += 1




def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
