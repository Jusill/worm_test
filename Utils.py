WORM_LENGTH = 2 #                      <-- INTERNAL PARAMETERS!
ORIENTATIONS = { #                     < DO NOT TOUCH FROM THIS
    'top' : 0, #                       <
    'right' : 1, #                     <
    'bottom' : 2, #                    <
    'left' : 3} #                      <
MOVEMENT = { #                         <
    0 : 'forward-left-attack', #       <
    1 : 'forward-left-noattack', #     <
    2 : 'forward-noturn-attack', #     <
    3 : 'forward-noturn-noattack', #   <
    4 : 'forward-right-attack', #      <
    5 : 'forward-right-noattack', #    <
    6 : 'nomove-left-attack', #        <
    7 : 'nomove-left-noattack', #      <
    8 : 'nomove-noturn-attack', #      <
    9 : 'nomove-noturn-noattack', #    <
    10 : 'nomove-right-attack', #      <
    11 : 'nomove-right-noattack', #    <
    12 : 'backward-left-attack', #     <
    13 : 'backward-left-noattack', #   <
    14 : 'backward-noturn-attack', #   <
    15 : 'backward-noturn-noattack', # <
    16 : 'backward-right-attack', #    <
    17 : 'backward-right-noattack' #   <
} #                                    <
VIEW = { #                             <
    'forward' : 5, #                   <
    'backward' : 5, #                  <
    'left' : 5, #                      <
    'right' : 5} #                     <-- TO THIS

WORM_MEMORY_SIZE = 30 #   <-- LEARNING PARAMETERS!
WORM_RECURRENT_VIEW = 5 # <
INITIAL_LR = 0.01 #       <   Initial learning rate, try different values
LEARN_BATCH_SIZE = 10 #   <   We learn on random batch from worm memory, this is max batch size
HEALTH_COEF = 0.5 #       <   REWARD FORMULA IS:
SATURATION_COEF = 0.5 #   <   reward = HEALTH_COEF*delta_health + SATURATION_COEF*delta_saturation + BREEDING_COEF*int(self.bred)
BREEDING_COEF = 0.5 #     <
AGE_ACTIVITY = 10 #       <-- lr = (INITIAL_LR  - reward/float(100))*float(AGE_ACTIVITY)/((1 + (self.time**2))*global_tick)

# HYPERPARAMETERS
# From here onwards can be changed freely
FOOD_RESTORATION = 100 # Food restores health and fully restores saturation
SPIKE_DAMAGE = 20 # Spike reduces health
SPIKE_DAMAGE_AOE = 2 # Spike hits in aoe (l1 metrics)
WORM_DAMAGE = 20 # Worm reduces other worm health if attacks
STARVATION_DAMAGE_THRESHOLD = 10
STARVATION_DAMAGE = 1
SATURATION_HEAL_THRESHOLD = 70
SATURATION_HEAL = 10
SATURATION_TICK_REDUCTION = 4
RENDER_DELAY = 50 # Render delay when visual debug show ms

# DEFAULT PARAMS MANIPULATION
thread_params = { # You can set the params here directly
    'world_name' : 'DEFAULT',
    'worms_init_number' : 100,
    'food_init_number' : 500,
    'spike_init_number' : 5,
    'spike_spawn_time' : 100,
    'food_spawn_time' : 50,
    'worm_spawn_time' : 10000,
    'spike_spawn_amount' : 5,
    'food_spawn_amount' : 500,
    'worm_spawn_amount' : 0,
    'world_width' : 100,
    'world_height' : 100,
    'learning' : True,
    'breeding' : True,
    'immortal' : False,
    'breeding_age' : 3,
    'breeding_prob' : 0.75,
    'breed_sat_share' : 0.2,
    'breed_sat_barrier' : 10,
    'world_lifespan' : 1000,
    'spike_lifespan' : 1000,
    'food_lifespan' : 1000,
    'worm_lifespan' : 100,
    'worm_speed' : 3,
    'worm_adequacy' : 0.7,
    'adequacy_increase_span' : 0.35,
    'global_adequacy_span' : 0.5,
    'learn_freq' : 6,
    'visual_width_scale' : 8,
    'visual_height_scale' : 8,
    'visual_debug_show' : True,
    'visual_save_recap' : True,
    'visual_worm_tribe_first_draw_color' : (255, 0, 0),
    'visual_worm_tribe_second_draw_color': (255, 0, 255),
    'visual_spike_draw_color' : (0, 0, 255),
    'visual_food_draw_color' : (0, 255, 0),
    'visual_fps' : 5}

# DESCRIPTION TABLE
cmd_params = { # Or you can start Thread.py with parameters from this list, the second string is the description (all flags start with --)
    'world-name=' : 'Name of the world configuration (string)',
    'worms-init-num=' : 'Initial amount of worms (int)',
    'food-init-num=' : 'Initial amount of food (int)',
    'spike-init-num=' : 'Initial amount of spikes (int)',
    'spike-spawn-time=' : 'Time for spawn new spikes (int)',
    'food-spawn-time=' : 'Time for spawn new food (int)',
    'worm-spawn-time=' : 'Time for spawn new worms (int)',
    'spike-spawn-amount=' : 'Amount of new spikes to be spawned (int)',
    'food-spawn-amount=' : 'Amount of new food to be spawned (int)',
    'worm-spawn-amount=' : 'Amount of new worms to be spawned (int)',
    'world-width=' : 'Width of the world (int)',
    'world-height=' : 'Height of the world (int)',
    'learning' : 'Learning',
    'breeding' : 'Breeding',
    'immortal' : 'Immortal worms',
    'breeding-age=' : 'Breeding age (int)',
    'breeding-prob=' : 'Probability of breeding when met (float, from 0 to 1)',
    'breeding-saturation-share=' : 'Part of saturation granted to child (float, from 0 to 1)',
    'breeding-saturation-barrier=' : 'Saturation level from which worm cat breed (int)',
    'world-lifespan=' : 'Lifespan of the world (int)',
    'spike-lifespan=' : 'Lifespan of spike (int)',
    'food-lifespan=' : 'Lifespan of food (int)',
    'worm-lifespan=' : 'Lifespan of the worm (int)',
    'worm-speed=' : 'Speed of worm (int)',
    'worm-adequacy=' : 'Adequacy of worm (chance to do as worm decided) (float, from 0 to 1)',
    'adequacy-increase-span=' : 'Period of worm lifespan within which the worm gets completely adequate (float, from 0 to 1)',
    'global-adequacy-span=' : 'Period of world lifespan within which worms store their adequacy (float, from 0 to 1)',
    'learn-frequency=' : 'Frequency of learning sessions (less than worm lifespan, more than LEARN_BATCH_SIZE) (int)',
    'visual-width-scale=' : 'Scale of visual width (int)',
    'visual-height-scale=' : 'Scale of visual height (int)',
    'visual-debug-show' : 'Do debug visual',
    'visual-save-recap' : 'Do save video recap',
    'visual-fps=' : 'FPS rate in video recap (int)'}

cmd_to_thread = { # INTERNAL USAGE ONLY
    'world-name' : 'world_name',
    'worms-init-num' : 'worms_init_number',
    'food-init-num' : 'food_init_number',
    'spike-init-num' : 'spike_init_number',
    'spike-spawn-time' : 'spike_spawn_time',
    'food-spawn-time' : 'food_spawn_time',
    'worm-spawn-time' : 'worm_spawn_time',
    'spike-spawn-amount' : 'spike_spawn_amount',
    'food-spawn-amount' : 'food_spawn_amount',
    'worm-spawn-amount' : 'worm_spawn_amount',
    'world-width' : 'world_width',
    'world-height' : 'world_height',
    'learning' : 'learning',
    'breeding' : 'breeding',
    'immortal' : 'immortal',
    'breeding-age' : 'breeding_age',
    'breeding-prob' : 'breeding_prob',
    'breeding-saturation-share' : 'breed_sat_share',
    'breeding-saturation-barrier' : 'breed_sat_barrier',
    'world-lifespan' : 'world_lifespan',
    'spike-lifespan' : 'spike_lifespan',
    'food-lifespan' : 'food_lifespan',
    'worm-lifespan' : 'worm_lifespan',
    'worm-speed' : 'worm_speed',
    'worm-adequacy' : 'worm_adequacy',
    'adequacy-increase-span' : 'adequacy_increase_span',
    'global-adequacy-span' : 'global_adequacy_span',
    'learn-frequency' : 'learn_freq',
    'visual-width-scale' : 'visual_width_scale',
    'visual-height-scale' : 'visual_height_scale',
    'visual-debug-show' : 'visual_debug_show',
    'visual-save-recap' : 'visual_save_recap',
    'visual-fps' : 'visual_fps'}
