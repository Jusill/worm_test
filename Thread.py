import sys, getopt
import numpy as np
import math
import numpy.random as npr
from time import sleep
import random

from Food import *
from Spike import *
from Environment import *

from Worm import *
from Colony import *

from Utils import *

from Visual import *

class Thread:
    def __init__(self, params):
        self.params = params
        self.params.update({'tick' : 0})

        env_max_times = {'spike' : params['spike_lifespan'], 'food' : params['food_lifespan']}
        self.environment = Environment(env_max_times)

        self.colony = Colony(params['worm_lifespan'])

        visual_params = {
            'world_width' : self.params['world_width'],
            'world_height' : self.params['world_height'],
            'width_scale' : self.params['visual_width_scale'],
            'height_scale' : self.params['visual_height_scale'],
            'worm_draw_color' : self.params['visual_worm_draw_color'],
            'spike_draw_color' : self.params['visual_spike_draw_color'],
            'food_draw_color' : self.params['visual_food_draw_color'],
            'fps' : self.params['visual_fps'],
            'debug_show' : self.params['visual_debug_show'],
            'save_recap' : self.params['visual_save_recap']}
        self.visual = Visual(visual_params)

    def _generate_init_worms(self):
        worms_params_x = npr.randint(0, self.params['world_width'], self.params['worms_init_number'])
        worms_params_y = npr.randint(0, self.params['world_height'], self.params['worms_init_number'])
        worms_params_orient = npr.randint(0, 4, self.params['worms_init_number'])
        for _ in range(self.params['worms_init_number']):
            x = worms_params_x[_]
            y = worms_params_y[_]
            orient = worms_params_orient[_]
            self.colony.emplace_worm(x, y, orient)

    def _generate_init_spikes(self):
        spike_params_x = npr.randint(0, self.params['world_width'], self.params['spike_init_number'])
        spike_params_y = npr.randint(0, self.params['world_height'], self.params['spike_init_number'])
        for _ in range(self.params['spike_init_number']):
            x = spike_params_x[_]
            y = spike_params_y[_]
            self.environment.emplace_spike(x, y)

    def _generate_init_food(self):
        food_params_x = npr.randint(0, self.params['world_width'], self.params['food_init_number'])
        food_params_y = npr.randint(0, self.params['world_height'], self.params['food_init_number'])
        for _ in range(self.params['food_init_number']):
            x = food_params_x[_]
            y = food_params_y[_]
            self.environment.emplace_food(x, y)

    def _tick(self):
        print('TIME: %d' % self.params['tick'])
        self.params['tick'] += 1
        self.environment.tick()
        self.colony.tick()
        self.stats = {
            'time' : self.params['tick'],
            'breedings' : 0,
            'crazy_actions' : 0,
            'attacks' : 0,
            'deaths' : 0,
            'resources_exhaustion' : 0,
            'population' : len(self.colony),
            'spikes_amount' : self.environment.spike_len(),
            'food_amount' : self.environment.food_len(),
            'world_lifespan' : self.params['world_lifespan'],
            'food_eaten' : 0,
            'spikes_hit' : 0,
            'food_spawned' : 0,
            'spikes_spawned' : 0,
            'loss' : 0.
        }

    def _is_alive(self):
        if len(self.colony) == 0:
            return False
        if self.params['tick'] <= self.params['world_lifespan']:
            return True
        else:
            return False

    def _render_world(self):
        world = np.zeros((self.params['world_width'], self.params['world_height'], 3), dtype=float)
        for x in range(self.params['world_width']):
            for y in range(self.params['world_height']):
                env_interception = self.environment.interception(x, y)
                colony_interception = self.colony.interception(x, y)
                world[x, y, 0] = colony_interception
                world[x, y, 1] = env_interception[0]
                world[x, y, 2] = env_interception[1]
        return world

    def _get_worm_view(self, worm_position):
        worm_x = worm_position[0]
        worm_y = worm_position[1]
        worm_orient = worm_position[2]
        vb = {
            'x1' : worm_x,
            'x2' : worm_x,
            'y1' : worm_y,
            'y2' : worm_y
        }
        if worm_orient == ORIENTATIONS['top']:
            vb['x1'] -= VIEW['left']
            vb['x2'] += VIEW['right'] + 1
            vb['y1'] -= VIEW['forward'] + WORM_LENGTH - 1
            vb['y2'] += VIEW['backward'] + 1
        elif worm_orient == ORIENTATIONS['bottom']:
            vb['x1'] -= VIEW['right']
            vb['x2'] += VIEW['left'] + 1
            vb['y1'] -= VIEW['backward']
            vb['y2'] += VIEW['forward'] + WORM_LENGTH
        elif worm_orient == ORIENTATIONS['left']:
            vb['x1'] -= VIEW['forward'] + WORM_LENGTH - 1
            vb['x2'] += VIEW['backward'] + 1
            vb['y1'] -= VIEW['right']
            vb['y2'] += VIEW['left'] + 1
        elif worm_orient == ORIENTATIONS['right']:
            vb['x1'] -= VIEW['backward']
            vb['x2'] += VIEW['forward'] + WORM_LENGTH
            vb['y1'] -= VIEW['left']
            vb['y2'] += VIEW['right'] + 1
        else:
            return None

        self.view_width = int(math.fabs(vb['x1'] - vb['x2']))
        self.view_height = int(math.fabs(vb['y1'] - vb['y2']))

        vb['x1'] %= self.params['world_width']
        vb['x2'] %= self.params['world_width']
        vb['x3'] = vb['x1']
        vb['x4'] = vb['x2']
        vb['y1'] %= self.params['world_height']
        vb['y2'] %= self.params['world_height']
        vb['y3'] = vb['y2']
        vb['y4'] = vb['y1']

        return vb

    def _rotate_worm_view(self, worm_view, orientation):
        new_worm_view = worm_view
        if orientation > 0:
            for i in range(orientation):
                new_worm_view = np.rot90(new_worm_view, axes=(1, 0))
        return new_worm_view

    def _normalize_worm_view(self, worm_view):
        new_worm_view = worm_view
        for i in range(3):
            mean = worm_view[:,:,i].mean()
            std = worm_view[:,:,i].std()
            new_worm_view[:,:,i] -= mean
            if std > 0:
                new_worm_view[:,:,i] /= std
        return new_worm_view

    def _update_worm_position(self, position, movement):
        move = movement[0]
        turn = movement[1]
        new_position = position
        if turn == 0: # turn left
            new_position[2] -= 1
            new_position[2] %= 4
        elif turn == 2: # turn right
            new_position[2] += 1
            new_position[2] %= 4
        move_value = 0
        if move == 0: # move forward
            move_value = self.params['worm_speed']
        elif move == 2: # move backward
            move_value = -self.params['worm_speed']
        # actual moving
        if new_position[2] == ORIENTATIONS['top']:
            new_position[1] -= move_value
        elif new_position[2] == ORIENTATIONS['bottom']:
            new_position[1] += move_value
        elif new_position[2] == ORIENTATIONS['left']:
            new_position[0] -= move_value
        elif new_position[2] == ORIENTATIONS['right']:
            new_position[0] += move_value

        new_position[0] %= self.params['world_width']
        new_position[1] %= self.params['world_height']
        return new_position

    def _colony_interaction(self, worm, attack):
        worm_position = worm.get_position()
        worm_x, worm_y = worm_position[0], worm_position[1]
        int_worm = None
        if self.world_view[worm_x, worm_y, 0] > 0:
            # TODO: Interaction with tails (they live on sphere)
            int_worm = self.colony.get_worm_by_position(worm_x, worm_y, except_for=worm.get_id())
        if int_worm:
            if attack == 0: # if worm decided to attack
                old_health = int_worm.get_health()
                new_health = old_health - WORM_DAMAGE
                int_worm.set_health(new_health)
                self.stats['attacks'] += 1

    def _food_interaction(self, worm, attack):
        worm_position = worm.get_position()
        worm_x, worm_y = worm_position[0], worm_position[1]
        int_food = None
        # if self.world_view[worm_x, worm_y, 2] > 0:
        int_food = self.environment.get_food_by_position(worm_x, worm_y)
        if int_food:
            restore = int_food.eat()
            old_health = worm.get_health()
            new_health = old_health + restore
            worm.set_health(new_health)
            worm.set_saturation(100.)
            self.stats['food_eaten'] += 1
            if attack == 0: # if worm decided to attack
                int_food.eat() # food gets double hit
                self.stats['food_eaten'] += 1

    def _spike_interaction(self, worm, attack):
        worm_position = worm.get_position()
        worm_x, worm_y = worm_position[0], worm_position[1]
        int_spike = None
        # if self.world_view[worm_x, worm_y, 1] > 0:
        int_spike = self.environment.get_spike_by_position(worm_x, worm_y)
        if int_spike:
            int_spike.hit()
            old_health = worm.get_health()
            new_health = old_health - SPIKE_DAMAGE
            worm.set_health(new_health)
            self.stats['spikes_hit'] += 1
            if attack == 0: # if worm decided to attack
                int_spike.hit() # spike gets double hit

    def _environment_interaction(self, worm, attack):
        self._spike_interaction(worm, attack)
        self._food_interaction(worm, attack)

    def _spawn_spike(self):
        if self.params['tick'] % self.params['spike_spawn_time'] == 0:
            for _ in range(self.params['spike_spawn_amount']):
                x_pos = npr.randint(0, self.params['world_width'])
                y_pos = npr.randint(0, self.params['world_height'])
                self.environment.emplace_spike(x_pos, y_pos)
                self.stats['spikes_spawned'] += 1

    def _spawn_food(self):
        if self.params['tick'] % self.params['food_spawn_time'] == 0:
            for _ in range(self.params['food_spawn_amount']):
                x_pos = npr.randint(0, self.params['world_width'])
                y_pos = npr.randint(0, self.params['world_height'])
                self.environment.emplace_food(x_pos, y_pos)
                self.stats['food_spawned'] += 1

    def _spawn_worm(self):
        if self.params['tick'] % self.params['worm_spawn_time'] == 0:
            for _ in range(self.params['worm_spawn_amount']):
                x_pos = npr.randint(0, self.params['world_width'])
                y_pos = npr.randint(0, self.params['world_height'])
                orient = npr.randint(0, 4)
                self.colony.emplace_worm(x_pos, y_pos, orient)

    def _spawn(self):
        self._spawn_spike()
        self._spawn_food()
        self._spawn_worm()

    def __fill_worm_view(self, vb):

        final_world_view = np.zeros((self.view_width, self.view_height, 3))

        if vb['x1'] < vb['x2']:
            if vb['y1'] < vb['y2']:
                final_world_view = self.world_view[vb['x1']:vb['x2'], vb['y1']:vb['y2'], :].copy()
            else:
                final_world_view[0:self.view_width, 0:self.params['world_height']-vb['y1'], :]\
                    = self.world_view[vb['x1']:vb['x4'],vb['y1']:self.params['world_height'], :].copy()

                final_world_view[0:self.view_width, self.params['world_height']-vb['y1']:self.view_height, :]\
                    = self.world_view[vb['x3']:vb['x4'], 0:vb['y3'], :].copy()
        else:
            if vb['y1'] < vb['y2']:
                final_world_view[0:self.params['world_width']-vb['x1'], 0:self.view_height, :] \
                    = self.world_view[vb['x1']:self.params['world_width'], vb['y1']:vb['y3'], :].copy()

                final_world_view[self.params['world_width']-vb['x1']:self.view_width, 0:self.view_height, :] \
                    = self.world_view[0:vb['x2'], vb['y4']:vb['y2'], :].copy()
            else:
                final_world_view[self.view_width-vb['x2']:self.view_width, self.view_height-vb['y2']:self.view_height, :] \
                    = self.world_view[0:vb['x2'], 0:vb['y2'], :].copy()

                final_world_view[0:self.view_width-vb['x2'], self.view_height-vb['y2']:self.view_height, :] \
                    = self.world_view[vb['x3']:self.params['world_width'], 0:vb['y3'], :].copy()

                final_world_view[self.view_width-vb['x2']:self.view_width, 0:self.view_height-vb['y2'], :] \
                    = self.world_view[0:vb['x4'], vb['y4']:self.params['world_height'], :].copy()

                final_world_view[0:self.view_width-vb['x2'], 0:self.view_height-vb['y2'], :] \
                    = self.world_view[vb['x1']:self.params['world_width'], vb['y1']:self.params['world_height'], :].copy()

        return final_world_view

    def _epsilon_rand(self, action, age):
        odds = npr.uniform(0, 1)
        inadequacy_cap = self.params['worm_lifespan']*self.params['adequacy_increase_span']
        worm_expirience = (1 - self.params['worm_adequacy'])*(float(age)/inadequacy_cap)
        global_inadequacy = max(1. - self.params['tick']/float(self.params['global_adequacy_span']*self.params['world_lifespan']), 0.)
        worm_adequacy = self.params['worm_adequacy'] + worm_expirience - global_inadequacy
        if odds > worm_adequacy: # time for crazy actions
            crazy_action = npr.randint(0, 18)
            self.stats['crazy_actions'] += 1
            return crazy_action
        return action

    def _extract_actions(self, action):
        move = action // 6
        turn = (action % 6) // 2
        attack = action % 2
        return move, turn, attack

    def _learn(self):
        if self.params['tick'] % self.params['learn_freq'] == 0:
            for worm in self.colony:
                self.stats['loss'] += worm.learn(self.params['tick'])
            self.stats['loss'] /= len(self.colony)

    def _breed(self, worm):
        if worm.get_time() < self.params['breeding_age'] or worm.did_bred() or worm.get_saturation() < self.params['breed_sat_barrier']:
            return
        x0, y0, or0 = worm.get_position()
        breed = self.colony.get_worm_by_position(x0, y0, except_for=worm.get_id())
        if breed == None:
            return
        if breed.get_time() < self.params['breeding_age'] or breed.did_bred() or breed.get_saturation() < self.params['breed_sat_barrier']:
            return
        while True:
            odds = npr.uniform(0, 1)
            if odds > self.params['breeding_prob']:
                return

            sd1 = worm.get_state_dict()
            sd2 = breed.get_state_dict()
            nsd = {}


            m_f = random.randint(0, 1)
            if m_f == 0:
                for k in sd1:
                    l = npr.uniform(-0.2, 0.2)
                    new_weight = (l*sd1[k]).clone().detach().requires_grad_(True)
                    nsd[k] = new_weight
            else:
                for k in sd2:
                    l = npr.uniform(-0.2, 0.2)
                    new_weight = (l * sd2[k]).clone().detach().requires_grad_(True)
                    nsd[k] = new_weight

            '''
            for k in sd1:
                l = npr.uniform()
                new_weight = (l*sd1[k] + (1 - l)*sd2[k]).clone().detach().requires_grad_(True)
                nsd[k] = new_weight
            '''

            x = npr.randint(x0 - 10, x0 + 10)
            y = npr.randint(y0 - 10, y0 + 10)
            orient = npr.randint(0, 4)
            sat1 = worm.get_saturation()
            sat2 = breed.get_saturation()
            new_sat = int(sat1*self.params['breed_sat_share']) + int(sat2*self.params['breed_sat_share'])
            worm.set_saturation(sat1 - int(sat1*self.params['breed_sat_share']))
            breed.set_saturation(sat2 - int(sat2*self.params['breed_sat_share']))
            self.colony.emplace_worm(x, y, orient, nsd, new_sat)
            worm.breed_restore()
            breed.breed_restore()
            self.stats['breedings'] += 1

    def _run(self):
        while(self._is_alive()):
            self._tick()
            self.world_view = self._render_world()
            for worm in self.colony:
                worm_position = list(worm.get_position())
                vb = self._get_worm_view(worm_position)
                worm_view = self.__fill_worm_view(vb)
                worm_view = self._rotate_worm_view(worm_view, worm_position[2])
                worm_view = self._normalize_worm_view(worm_view)
                action = worm(worm_view) # feed worm view to worm
                action = self._epsilon_rand(action, worm.get_time())
                move, turn, attack = self._extract_actions(action)
                print(move, turn, attack)
                movement = (move, turn)
                worm_position = self._update_worm_position(worm_position, movement)
                worm.set_position(*worm_position)
                self._colony_interaction(worm, attack)
                self._environment_interaction(worm, attack)
            for worm in self.colony:
                if self.params['breeding']:
                    self._breed(worm)
                worm.restore()
                worm.memorize()

            if self.params['learning']:
                self._learn()

            if not self.params['immortal']:
                self.stats['deaths'] = self.colony.clean_up()

            self.stats['resources_exhaustion'] = self.environment.clean_up()
            self._spawn()
            self.visual.show(self.colony, self.environment, self.stats)

            if self.params['visual_debug_show']:
                sleep(RENDER_DELAY*(10**(-3)))
            else:
                health_distribution = [0 for i in range(101)]
                saturation_distribution = [0 for i in range(101)]
                age_distribution = [0 for i in range(self.colony.max_time + 1)]
                print(self.colony.len())
                for w in self.colony:
                    w_health = int(round(w.get_health()))
                    w_saturation = int(round(w.get_saturation()))
                    w_age = w.get_time()
                    if w_health >= 0:
                        health_distribution[w_health] += 1
                    if w_saturation >= 0:
                        saturation_distribution[w_saturation] += 1
                    if w_age >= 0:
                        age_distribution[min(w_age, self.colony.max_time)] += 1
                print("HEALTH: ", health_distribution)
                print("SATURATION: ", saturation_distribution)
                print("AGE: ", age_distribution)
        self.visual.clear()

    def generate(self):
        self._generate_init_worms()
        self._generate_init_spikes()
        self._generate_init_food()

    def show_params(self):
        hparams = self.params.copy()
        hparams['WORM_LENGTH'] = WORM_LENGTH
        hparams['MEMORY_SIZE'] = WORM_MEMORY_SIZE
        hparams['WORM_RECURRENT_VIEW'] = WORM_RECURRENT_VIEW
        hparams['INITIAL_LR'] = INITIAL_LR
        hparams['LEARN_BATCH_SIZE'] = LEARN_BATCH_SIZE
        hparams['HEALTH_COEF'] = HEALTH_COEF
        hparams['SATURATION_COEF'] = SATURATION_COEF
        hparams['BREEDING_COEF'] = BREEDING_COEF
        hparams['AGE_ACTIVITY'] = AGE_ACTIVITY
        hparams['FOOD_RESTORATION'] = FOOD_RESTORATION
        hparams['SPIKE_DAMAGE'] = SPIKE_DAMAGE
        hparams['SPIKE_DAMAGE_AOE'] = SPIKE_DAMAGE_AOE
        hparams['WORM_DAMAGE'] = WORM_DAMAGE
        hparams['STARVATION_DAMAGE_THRESHOLD'] = STARVATION_DAMAGE_THRESHOLD
        hparams['STARVATION_DAMAGE'] = STARVATION_DAMAGE
        hparams['SATURATION_HEAL_THRESHOLD'] = SATURATION_HEAL_THRESHOLD
        hparams['SATURATION_HEAL'] = SATURATION_HEAL
        hparams['SATURATION_TICK_REDUCTION'] = SATURATION_TICK_REDUCTION
        hparams['RENDER_DELAY'] = RENDER_DELAY
        self.visual.show_params(hparams)

    def start(self):
        self._run()


if __name__ == "__main__":
    opts, _ = getopt.getopt(sys.argv[1:], "", cmd_params.keys())
    for key, value in opts:
        if key[2:] in cmd_to_thread.keys():
            param_name = cmd_to_thread[key[2:]]
            if param_name == 'world_name':
                thread_params[param_name] = value
                print('-< %s has been set to %s\n' % (cmd_params[key[2:] + "="], str(value)))
            elif value:
                thread_params[param_name] = int(value)
                print('-< %s has been set to %s\n' % (cmd_params[key[2:] + "="], str(value)))
            else:
                thread_params[param_name] = True
                print('-< %s has been set to True\n' % (cmd_params[key[2:]]))
    main = Thread(thread_params)
    main.generate()
    main.show_params()
    main.start()
