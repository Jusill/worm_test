import cv2
import numpy as np
import time

from Utils import *

from Food import Food
from Spike import Spike
from Worm import Worm

class Visual:
    def __init__(self, params):
        self.params = params
        self.width = self.params['world_width']*self.params['width_scale']
        self.height = self.params['world_height']*self.params['height_scale']
        if self.params['debug_show']:
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Frame', self.width, self.height)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if self.params['save_recap']:
            self.out = cv2.VideoWriter('./recaps/%s.avi' % str(time.time()), fourcc, self.params['fps'], (3*self.width, 5*self.height))
        self.stats = {}
        self.long_stats = {}

    def show_params(self, hparams):
        self.params_frame = np.zeros((5*self.height, 3*self.width, 3), np.uint8)
        self.params_frame.fill(255)
        v_offset = 0.1*5*self.height
        h_offset = 0.1*3*self.width
        block_height = 0.8*2*3*self.height/float(len(hparams))
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 2#self.height/float(6*110)
        cv2.putText(self.params_frame, hparams['world_name'], (int(0.5*3*self.width - h_offset), int(v_offset*0.5)),
                        font, 2*font_size, (140, 32, 64), 2, cv2.LINE_AA)
        for i, param_name in enumerate(hparams):
            if i % 2 == 0:
                cv2.putText(self.params_frame, '%s = %s' % (param_name, str(hparams[param_name])),
                                            (int(h_offset), int(v_offset + i*block_height)),
                                            font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(self.params_frame, '%s = %s' % (param_name, str(hparams[param_name])),
                                            (int(0.5*3*self.width + h_offset), int(v_offset + (i-1)*block_height)),
                                            font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        if self.params['save_recap']:
            for i in range(self.params['fps']*2):
                self.out.write(self.params_frame)
        if self.params['debug_show']:
            cv2.imshow('Frame', self.params_frame)
            cv2.waitKey(10*RENDER_DELAY)


    def show(self, colony, environment, stats):
        self.colony = colony
        self.environment = environment
        self.stats = stats
        for key in self.stats:
            if key in self.long_stats.keys():
                self.long_stats[key].append(self.stats[key])
            else:
                self.long_stats[key] = []
        self.frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.frame.fill(255)
        self.show_positions()
        self.show_stats()
        self.show_graphs()
        if self.params['save_recap']:
            self.out.write(self.frame)
        if self.params['debug_show']:
            cv2.imshow('Frame', self.frame)
            cv2.waitKey(RENDER_DELAY)

    def show_positions(self):
        for el in self.environment:
            el_position = el.get_position()
            el_x, el_y = el_position[:2]
            x1 = el_x*self.params['width_scale']
            x2 = (el_x+1)*self.params['width_scale']
            y1 = el_y*self.params['height_scale']
            y2 = (el_y+1)*self.params['height_scale']
            draw_color = self.params['spike_draw_color']
            if isinstance(el, Food):
                draw_color = self.params['food_draw_color']
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), draw_color, -1)
        for w in self.colony:
            worm_position = w.get_position()
            worm_x, worm_y, orient = worm_position[:]
            x1 = worm_x*self.params['width_scale']
            x2 = (worm_x+1)*self.params['width_scale']
            y1 = worm_y*self.params['height_scale']
            y2 = (worm_y+1)*self.params['height_scale']
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), self.params['worm_draw_color'], -1)

    def show_stats(self):
        self.stats_frame = np.zeros((self.height, 2*self.width, 3), np.uint8)
        self.stats_frame.fill(255)
        cv2.line(self.stats_frame, (1, 0), (1, self.height),
                                    (0, 0, 0), 3)
        health_distribution = [0 for i in range(101)]
        saturation_distribution = [0 for i in range(101)]
        age_distribution = [0 for i in range(self.colony.max_time + 1)]
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
        health_frame = {
            'name' : 'Health distribution',
            'y1' : 0,
            'y2' : self.height // 4,
            'max_value' : 100,
            'min_value' : 0
        }
        saturation_frame = {
            'name' : 'Saturation distribution',
            'y1' : self.height // 4,
            'y2' : 2*self.height // 4,
            'max_value' : 100,
            'min_value' : 0
        }
        age_frame = {
            'name' : 'Age distribution',
            'y1' : 2*self.height // 4,
            'y2' : 3*self.height // 4,
            'max_value' : self.colony.max_time,
            'min_value' : 0
        }
        info_frame = {
            'name' : 'Statistics',
            'y1' : 3*self.height // 4,
            'y2' : self.height
        }
        self.show_distribution(health_distribution, health_frame)
        self.show_distribution(saturation_distribution, saturation_frame)
        self.show_distribution(age_distribution, age_frame)
        self.show_info(info_frame)
        self.frame = np.concatenate((self.frame, self.stats_frame), axis=1)

    def show_distribution(self, distribution, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        block_height = frame['y2'] - frame['y1']
        block_width = 1.95*self.width/float(len(distribution))
        font_size = block_height/float(2*110)
        title = int(block_height * 0.1)
        body = int(block_height * 0.6)
        cv2.putText(self.stats_frame, frame['name'], (int(0.05*2*self.width), frame['y1'] + title),
                                                      font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.stats_frame, '%d' % frame['min_value'], (0, frame['y2'] - title),
                                            font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(self.stats_frame, '%d' % frame['max_value'], (int(0.95*2*self.width), frame['y2'] - title),
                                            font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        top_value = sum(distribution)
        if top_value == 0:
            cv2.rectangle(self.stats_frame, (int(block_width), frame['y2'] - 2*title),
                                            (int(block_width*len(distribution)), frame['y2'] - 2*title),
                                            (0, 0, 0), 3)
        else:
            for horizontal, value in enumerate(distribution):
                height = int(body*float(value)/top_value)
                cv2.rectangle(self.stats_frame, (int((horizontal + 1)*block_width), frame['y2'] - 2*title),
                                                (int((horizontal + 2)*block_width), frame['y2'] - 2*title - height),
                                                (0, 0, 0), 3)

    def show_info(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        block_width = int(float(2*self.width) / (len(self.stats.keys()) / float(3)))
        block_height = (frame['y2'] - frame['y1']) // 3
        offset = int(0.025*2*self.width)
        font_size = block_height/float(110)
        for i, stat_name in enumerate(self.stats):
            stat_value = self.stats[stat_name]
            stat_string = '%s : %d' % (stat_name, stat_value)
            horizontal = i // 3
            vertical = i % 3
            cv2.putText(self.stats_frame, stat_string, (offset + block_width*horizontal,
                                                        frame['y1'] + int(block_height*(vertical + 0.5))),
                                                        font, font_size, (0, 0, 0), 2, cv2.LINE_AA)

    def show_graphs(self):
        if self.stats['time'] == 1:
            return
        self.graphs_frame = np.zeros((4*self.height, 3*self.width, 3), np.uint8)
        self.graphs_frame.fill(255)
        bd_frame = {
            'name' : 'Birth-death-population',
            'y1' : 0,
            'y2' : 4*self.height // 6,
            'names' : ['breedings', 'deaths', 'population'],
            'colors' : [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        }
        fs_frame = {
            'name' : 'Food-spike hits',
            'y1' : 4*self.height // 6,
            'y2' : 2*4*self.height // 6,
            'names' : ['food_eaten', 'spikes_hit'],
            'colors' : [self.params['food_draw_color'], self.params['spike_draw_color']]
        }
        res_frame = {
            'name' : 'Resources',
            'y1' : 2*4*self.height // 6,
            'y2' : 3*4*self.height // 6,
            'names' : ['food_amount', 'spikes_amount'],
            'colors' : [self.params['food_draw_color'], self.params['spike_draw_color']]
        }
        spawn_frame = {
            'name' : 'Food-spike spawns',
            'y1' : 3*4*self.height // 6,
            'y2' : 4*4*self.height // 6,
            'names' : ['food_spawned', 'spikes_spawned'],
            'colors' : [self.params['food_draw_color'], self.params['spike_draw_color']]
        }
        crazy_frame = {
            'name' : 'Crazy actions-attacks-population',
            'y1' : 4*4*self.height // 6,
            'y2' : 5*4*self.height // 6,
            'names' : ['crazy_actions', 'attacks', 'population'],
            'colors' : [(13, 92, 180), (0, 0, 255), (255, 0, 0)]
        }
        loss_frame = {
            'name' : 'Loss',
            'y1' : 5*4*self.height // 6,
            'y2' : 4*self.height,
            'names' : ['loss'],
            'colors' : [(0, 61, 150)]
        }
        self.show_graph(bd_frame)
        self.show_graph(fs_frame)
        self.show_graph(spawn_frame)
        self.show_graph(crazy_frame)
        self.show_graph(res_frame)
        self.show_graph(loss_frame)
        self.frame = np.concatenate((self.frame, self.graphs_frame), axis=0)

    def show_graph(self, frame):
        cv2.line(self.graphs_frame, (0, frame['y1'] + 1), (3*self.width, frame['y1'] + 1),
                                    (0, 0, 0), 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        block_height = int((frame['y2'] - frame['y1'])*0.8)
        block_width = 3*0.90*self.width/float(self.stats['world_lifespan'])
        offset = int(3*0.05*self.width)
        font_size = block_height/float(440)
        legend_width = 3*0.2*self.width
        rect_width = 3*0.05*self.width
        legend_block_height = block_height/float(len(frame['names']))
        bottom_offset = int((frame['y2'] - frame['y1'])*0.05)
        cv2.putText(self.graphs_frame, frame['name'], (int(3*0.1*self.width),
                                                    frame['y1'] + int(0.1*block_height)),
                                                    font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
        top = 0
        for i, name in enumerate(frame['names']):
            cv2.rectangle(self.graphs_frame, (int(3*self.width - legend_width),
                                               int(frame['y1'] + 0.1*(frame['y2'] - frame['y1']) + legend_block_height*i)),
                                             (int(3*self.width - legend_width + rect_width),
                                               int(frame['y1'] + 0.1*(frame['y2'] - frame['y1']) + legend_block_height*(i+1))),
                                             frame['colors'][i], -1)
            cv2.putText(self.graphs_frame, '  - %s' % name, (int(3*self.width - legend_width + rect_width),
                                                             int(frame['y1'] + 0.1*(frame['y2'] - frame['y1']) + legend_block_height*(i + 0.5))),
                                                             font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
            top = max(top, max(self.long_stats[name]))
        if top == 0:
            top = 100
        for draw_color, name in zip(frame['colors'], frame['names']):
            prev_value = self.long_stats[name][0]
            prev_horizontal = 0
            for new_horizontal, new_value in enumerate(self.long_stats[name]):
                x1 = int(offset + prev_horizontal*block_width)
                x2 = int(offset + new_horizontal*block_width)
                y1 = int(frame['y2'] - bottom_offset - (float(prev_value)/top)*block_height)
                y2 = int(frame['y2'] - bottom_offset - (float(new_value)/top)*block_height)
                cv2.line(self.graphs_frame, (x1, y1), (x2, y2), draw_color, 2)
                prev_value = new_value
                prev_horizontal = new_horizontal

    def clear(self):
        if self.params['save_recap']:
            self.out.release()
        cv2.destroyAllWindows()
