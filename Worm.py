from Nnet import WormNET
from Memory import Storage
from Utils import WORM_MEMORY_SIZE, WORM_RECURRENT_VIEW, INITIAL_LR, LEARN_BATCH_SIZE
from Utils import HEALTH_COEF, SATURATION_COEF, AGE_ACTIVITY
from Utils import SATURATION_TICK_REDUCTION, STARVATION_DAMAGE_THRESHOLD
from Utils import STARVATION_DAMAGE, SATURATION_HEAL_THRESHOLD, SATURATION_HEAL
from Utils import BREEDING_COEF
from math import fabs

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

class Worm:
    def __init__(self, id, x, y, orientation=0, weights=None):
        self.id = id
        self.health = 100.
        self.time = 0
        self.power = 1
        self.saturation = 100.
        self.orient = orientation
        self.bred = False

        self.net = WormNET()
        if weights:
            self.net.load_state_dict(weights)
        self.storage = Storage(WORM_MEMORY_SIZE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), INITIAL_LR)
        self.good_loss_fn = nn.CrossEntropyLoss()
        self.bad_loss_fn = nn.MSELoss()
        self.x = x
        self.y = y

    def __call__(self, env):
        feed_env = torch.from_numpy(env.copy())
        feed_env = feed_env.permute(2, 0, 1)
        feed_env = feed_env.view(1, 3, 11, 12)
        feed_env = feed_env.float()
        prev_views = self.storage.topn(WORM_RECURRENT_VIEW)

        if prev_views == None:
            prev_views = [feed_env for i in range(WORM_RECURRENT_VIEW)]
        else:
            prev_views = [prev_views[i][0] for i in range(WORM_RECURRENT_VIEW)]
        feed_view = tuple(prev_views + [feed_env])
        feed_view = torch.cat(feed_view, dim=1)

        with torch.no_grad():
            prob = self.net(feed_view)

        self._view = feed_env
        self._inputs = feed_view
        self._act = prob
        self._state = {
            'health': self.health,
            'saturation': self.saturation
        }
        return np.argmax(prob.numpy()[0, 0, 0, :])

    def memorize(self):
        if self.time > 0:
            self._state['health'] = self.health - self._state['health']
            self._state['saturation'] = self.saturation - self._state['saturation']
            self.storage.push_memo(self._view, self._inputs, self._act, self._state)

    def learn(self, global_tick):
        if self.time == 0:
            return 0.
        self.net.train()
        lbs = npr.randint(1, LEARN_BATCH_SIZE)
        learn_batch = self.storage.batch(lbs)
        ret_loss = 0.
        for view, feed_view, act, state in learn_batch:
            reward = HEALTH_COEF*state['health'] + SATURATION_COEF*state['saturation'] + BREEDING_COEF*int(self.bred)
            if reward <= 0:
                lr = (INITIAL_LR  - reward/float(1000))*(float(AGE_ACTIVITY)/self.time)*(0.1**(global_tick // 330))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                target = np.argmax(act.numpy()[0, 0, 0, :])
                tensor_target = torch.zeros([1])
                prob = self.net(feed_view)
                loss = self.bad_loss_fn(prob[0, 0, 0, target], tensor_target)
                ret_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                lr = max(INITIAL_LR  - reward/float(1000), 0.)*(float(AGE_ACTIVITY)/self.time)*(0.1**(global_tick // 330))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                target = np.argmax(act.numpy()[0, 0, 0, :])
                tensor_target = torch.ones([1], dtype=torch.long)
                tensor_target[0] = torch.as_tensor(target)
                prob = self.net(feed_view)
                loss = self.good_loss_fn(prob[0, 0, :, :], tensor_target)
                ret_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return ret_loss / lbs

    def get_id(self):
        return self.id

    def get_power(self):
        return self.power

    def set_power(self, p):
        self.power = p

    def get_health(self):
        return self.health

    def set_health(self, h):
        self.health = h
        self.health = max(min(self.health, 100), 0)

    def get_saturation(self):
        return self.saturation

    def set_saturation(self, s):
        self.saturation = s

    def get_time(self):
        return self.time

    def tick(self):
        self.time += 1
        self.bred = False

    def restore(self):
        self.saturation -= SATURATION_TICK_REDUCTION
        self.saturation = max(self.saturation, 0.)
        if self.saturation <= STARVATION_DAMAGE_THRESHOLD:
            self.health -= STARVATION_DAMAGE
            self.health = max(self.health, 0.)
        elif self.saturation >= SATURATION_HEAL_THRESHOLD:
            self.health += SATURATION_HEAL
            self.health = min(self.health, 100.)

    def breed_restore(self):
        self.bred = True

    def did_bred(self):
        return self.bred

    def get_position(self):
        return (self.x, self.y, self.orient)

    def set_position(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orient = orientation

    def get_state_dict(self):
        return self.net.state_dict()
