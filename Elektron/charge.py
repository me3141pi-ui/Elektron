import numpy as np

null_v = np.array([0,0,0])

class piq:
    def __init__(self,m = 1,q = 1,init_v=null_v[:],init_r=null_v[:],initial_acceleration=null_v[:]):
        #defining mass and charge
        self.mass = m
        self.charge = q
        #defining its initial position and velocity
        self.position = init_r
        self.velocity = init_v
        self.acceleration = initial_acceleration



