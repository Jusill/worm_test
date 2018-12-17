import numpy as np
import random
import copy


def get_best_worm(colony):
    worms = []

    for worm in colony:
        worm.set_loss_GA()

    for worm in colony:
        worms.append(copy.copy(worm))
    worms.sort(key=lambda worm: worm.loss_GA, reverse=True)

    #for worm in worms:
    #    print(worm.get_loss_GA())

    best_worms = worms[:5]

    best_worms.extend(list(np.random.choice(worms[5:len(worms)//3], 5, replace=False)))

    return best_worms


def mutate(worm):
    new_worm = {}
    weights = worm.get_state_dict()

    for i in weights:
        rand = np.random.choice([-0.05, 0, 0.05])
        weight = weights[i].clone().detach().requires_grad_(True) + rand
        new_worm[i] = weight

    return new_worm
