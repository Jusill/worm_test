from Utils import SPIKE_DAMAGE

class Spike:
    def __init__(self, id, x, y, strength=100):
        self.__id = id
        self.__strength = strength
        self.__power = SPIKE_DAMAGE
        self.__time = 0
        self.__x = x
        self.__y = y
    
    def tick(self):
        self.__time += 1
    
    def hit(self):
        self.__strength -= SPIKE_DAMAGE

        if self.__strength < 0:
            self.__strength = 0
        return self.__power
    
    def set_strenght(self, strenght):
        self.__strength = strenght
    
    def set_power(self, power):
        self.__power = power
    
    def get_time(self):
        return self.__time
    
    def get_id(self):
        return self.__id
    
    def get_strength(self):
        return self.__strength
    
    def get_power(self):
        return self.__power
    
    def get_position(self):
        return (self.__x, self.__y)

