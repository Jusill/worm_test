from Utils import FOOD_RESTORATION

class Food:
    def __init__(self, id, x, y, saturation=500):
        self.__id = id
        self.__saturation = saturation
        self.__time = 0
        self.__x = x
        self.__y = y

    def tick(self):
        self.__time += 1

    def eat(self):
        restore = FOOD_RESTORATION
        self.__saturation -= FOOD_RESTORATION

        if self.__saturation <= 0:
            restore += self.__saturation
            self.__saturation = 0
        return restore

    def set_saturation(self, saturation):
        self.__saturation = saturation

    def get_id(self):
        return self.__id

    def get_saturation(self):
        return self.__saturation

    def get_time(self):
        return self.__time

    def get_position(self):
        return (self.__x, self.__y)
