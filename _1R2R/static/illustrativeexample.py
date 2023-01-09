import numpy as np
import tensorflow as tf

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = (obs > 0.1)
        return done

    @staticmethod
    def termination_fn_tf(obs, act, next_obs):
        done = tf.cast(tf.greater(obs, 0.1), tf.float32)
        return done
