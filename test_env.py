import numpy as np
class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False
    high             = 255.

    # output config
    output_path  = ""
    model_output1 = output_path + "tmodel1.weights"
    model_output2 = output_path + "tmodel2.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 20
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 5000
    log_freq          = 50
    eval_freq         = 500
    soft_epsilon      = 0

    # hyper params
    nsteps_train       = 5000
    batch_size         = 32
    buffer_size        = 500
    target_update_freq = 500
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 1
    lr                 = 0.00015
    lr_begin           = 0.00025
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = nsteps_train/2
    learning_start     = 500

class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape
        self.state_0 = np.random.randint(0, 50, shape, dtype=np.int16)
        self.state_1 = np.random.randint(100, 150, shape, dtype=np.int16)
        self.state_2 = np.random.randint(200, 250, shape, dtype=np.int16)
        self.state_3 = np.random.randint(300, 350, shape, dtype=np.int16)
        self.states = [self.state_0, self.state_1, self.state_2, self.state_3]   


class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    Modified 
    """
    def __init__(self, shape=(84, 84, 3)):
        #4 states
        self.rewards = [0.2 , -0.1, 0.0, -0.3]
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        self.action_space = ActionSpace(5)
        self.observation_space = ObservationSpace(shape)
        

    def reset(self):
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        return self.observation_space.states[self.cur_state]
        

    def step(self, action):
        assert(0 <= action <= 4)
        self.num_iters += 1
        if action < 4:   
            self.cur_state = action
        reward = self.rewards[self.cur_state]
        if self.was_in_second is True:
            reward *= -10
        if self.cur_state == 2:
            self.was_in_second = True
        else:
            self.was_in_second = False
        return self.observation_space.states[self.cur_state], reward, self.num_iters >= 5, {'ale.lives':0}


    def render(self):
        print(self.cur_state)