import numpy as np
from . import charge
import matplotlib.pyplot as plt
import math

class piqSpace:
    def __init__(self, k=9 * (10 ** 9), c=3 * (10 ** 8)):
        self.charges = [] #list for holding the charges in the piqSpace

        #defining the constants for electric fields
        self.k = k
        self.epsilon = 1/(4*k*np.pi)
        #defining the constants for magnetic field
        self.mu = 1/((c**2)*(self.epsilon))
        self.km = self.mu/(4*np.pi)

    def add_charge(self,charge):
        if str(type(charge))== '<class \'Elektron.charge.piq\'>':
            self.charges.append(charge)
        else:
            raise TypeError("Charge must be of the type charge.piq")

    def getEfield(self,r):
        x,y,z = r
        Ex,Ey,Ez = 0,0,0
        for charge in self.charges:
            X,Y,Z = charge.position
            xd , yd , zd = x-X,y-Y,z-Z

            mag = (xd**2 + yd**2 + zd**2)**(1/2)
            if mag ==0:
                continue

            Ex += self.k*charge.charge*xd/mag**3
            Ey += self.k*charge.charge*yd/mag**3
            Ez += self.k*charge.charge*zd/mag**3
        return np.array([Ex,Ey,Ez])

    def getBfield(self,r):
        x, y, z = r
        Bx, By, Bz = 0, 0, 0
        for charge in self.charges:
            X,Y,Z = charge.position
            xd , yd , zd = x-X,y-Y,z-Z
            vx , vy , vz = charge.velocity

            mag = (xd**2 + yd**2 + zd**2)**(1/2)
            if mag ==0:
                continue

            Bx += self.km*(vy * zd - vz * yd)*charge.charge/mag**3
            By += self.km*(vz * xd - vx * zd)*charge.charge/mag**3
            Bz += self.km*(vx * yd - vy * xd)*charge.charge/mag**3
        return [Bx,By,Bz]

    #function to visualise the electrostatic field
    def esVisual(self,boundary_x=(-10,10),boundary_y=(-10,10),boundary_z=(-10,10),num=(10,10,10),distort = False,plot_charge = False,**kwargs):
        #Defining the X , Y and Z coordinates for all the points represented
        #NOTE : Its better to keep the length of x ,y ,z scales equal to prevent distortions
        if not distort:
            deviation = max(np.abs(boundary_x[1]-boundary_x[0]),np.abs(boundary_y[1]-boundary_y[0]),np.abs(boundary_z[1]-boundary_z[0]))/2
            centerx,centery,centerz = (boundary_x[0]+boundary_x[1])/2,(boundary_y[0]+boundary_y[1])/2,(boundary_z[0]+boundary_z[1])/2
            boundary_x = (centerx - deviation,centerx+deviation)
            boundary_y = (centery - deviation,centery+deviation)
            boundary_z = (centerz - deviation,centerz+deviation)

        xs = np.linspace(boundary_x[0], boundary_x[1], num[0])
        ys = np.linspace(boundary_y[0], boundary_y[1], num[1])
        zs = np.linspace(boundary_z[0], boundary_z[1], num[2])

        X,Y,Z = np.meshgrid(xs,ys,zs)

        Ex,Ey,Ez = 0 , 0 , 0
        #calculating the electric field for all the desired points
        for charge in self.charges:
            X_d,Y_d,Z_d = X-charge.position[0],Y-charge.position[1],Z-charge.position[2]
            r2 = X_d**2 + Y_d**2 + Z_d**2
            r2[r2 ==0] = np.inf

            Ex_d = self.k*charge.charge * X_d /r2**1.5
            Ey_d = self.k*charge.charge * Y_d / r2**1.5
            Ez_d = self.k*charge.charge * Z_d / r2**1.5

            Ex,Ey,Ez= Ex+Ex_d, Ey+Ey_d, Ez+Ez_d

        #defining the 3d graph to represent the vectors
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        if not distort:
            axis.set_box_aspect([1, 1, 1])
        axis.set_xlabel('X axis')
        axis.set_ylabel('Y axis')
        axis.set_zlabel('Z axis')

        #flattening the X,Y,Z and Ex,Ey,Ez np arrays for plotting
        #also calculating the magnitude array
        Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
        Exf, Eyf, Ezf = Ex.ravel(), Ey.ravel(), Ez.ravel()
        E_mag = (Exf ** 2 + Eyf ** 2 + Ezf ** 2) ** 0.5

        if 'mag_lim' in kwargs:
            map = E_mag <= kwargs['mag_lim']
        else:
            map = np.ones_like(E_mag)

        E_mag *= map #reducing all vectors whose magnitude exceeds the limit to 0
        #creating a normalization function for normalizing between E mag min and E mag max
        normalization = plt.Normalize((E_mag).min(), (E_mag).max())

        #Creating a color map to map vector magnitude to different colors (as per the gradient color scheme) using the normalization function above
        if 'cmap' in kwargs:
            try:
                colors = getattr(plt.cm, kwargs['cmap'])(normalization(E_mag))
            except:
                colors = plt.cm.plasma(normalization(E_mag))
        else:
            colors = 'black'

        if 'length' in kwargs and type(kwargs['length']) in [float, int]:
            l = kwargs['length']/E_mag.max()
        else:
            l = 1/E_mag.max()

        if plot_charge:
            Xq,Yq,Zq,colorQ,sizes  = [],[],[],[],[]
            if 'charge_size' in kwargs:
                scale = kwargs['charge_size']
            else:
                scale = 100
            for charge in self.charges:
                Xq.append(charge.position[0]),Yq.append(charge.position[1]),Zq.append(charge.position[2])
                colorQ.append('red' if charge.charge>=0 else 'blue')
                sizes.append(np.abs(np.abs(charge.charge)*scale))
            axis.scatter(Xq,Yq,Zq,color=colorQ,s=sizes)
        #finally, plotting the vector field
        axis.quiver(Xf, Yf, Zf, Exf * map, Eyf * map, Ezf * map, color=colors, length=l)
        plt.show()

    #visualising the magnetic fields for the given charges
    def magVisual(self, boundary_x=(-10, 10), boundary_y=(-10, 10), boundary_z=(-10, 10), num=(10, 10, 10),distort = False,plot_charge = False, **kwargs):
        #creating the vector space
        if not distort:
            deviation = max(np.abs(boundary_x[1]-boundary_x[0]),np.abs(boundary_y[1]-boundary_y[0]),np.abs(boundary_z[1]-boundary_z[0]))/2
            centerx,centery,centerz = (boundary_x[0]+boundary_x[1])/2,(boundary_y[0]+boundary_y[1])/2,(boundary_z[0]+boundary_z[1])/2
            boundary_x = (centerx - deviation,centerx+deviation)
            boundary_y = (centery - deviation,centery+deviation)
            boundary_z = (centerz - deviation,centerz+deviation)


        xs = np.linspace(boundary_x[0], boundary_x[1], num[0])
        ys = np.linspace(boundary_y[0], boundary_y[1], num[1])
        zs = np.linspace(boundary_z[0], boundary_z[1], num[2])

        X,Y,Z = np.meshgrid(xs,ys,zs)
        Bx, By, Bz = 0, 0, 0
        for charge in self.charges:
            rx, ry, rz = charge.position
            Xd, Yd, Zd = X - rx, Y - ry, Z - rz
            R_mag = (Xd ** 2 + Yd ** 2 + Zd ** 2) ** 0.5
            R_mag[R_mag == 0] = np.inf
            vx, vy, vz = charge.velocity

            bx = self.km * charge.charge * (vy * Zd - vz * Yd) / R_mag ** 3
            by = self.km * charge.charge * (vz * Xd - vx * Zd) / R_mag ** 3
            bz = self.km * charge.charge * (vx * Yd - vy * Xd) / R_mag ** 3
            Bx = Bx + bx
            By = By + by
            Bz = Bz + bz

        # defining the 3d graph to represent the vectors
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        if not distort:
            axis.set_box_aspect([1, 1, 1])
        axis.set_xlabel('X axis')
        axis.set_ylabel('Y axis')
        axis.set_zlabel('Z axis')

        # flattening the X,Y,Z and Bx,By,Bz np arrays for plotting
        # also calculating the magnitude array
        Xf, Yf, Zf = X.ravel(), Y.ravel(), Z.ravel()
        Bxf, Byf, Bzf = Bx.ravel(), By.ravel(), Bz.ravel()
        B_mag = (Bxf ** 2 + Byf ** 2 + Bxf ** 2) ** 0.5

        if 'mag_lim' in kwargs:
            map = B_mag <= kwargs['mag_lim']
        else:
            map = np.ones_like(B_mag)

        B_mag *= map #reducing all vectors whose magnitude exceeds the limit to 0
        #creating a normalization function for normalizing between E mag min and E mag max
        normalization = plt.Normalize((B_mag).min(), (B_mag).max())

        #Creating a color map to map vector magnitude to different colors (as per the gradient color scheme) using the normalization function above
        if 'cmap' in kwargs:
            try:
                colors = getattr(plt.cm, kwargs['cmap'])(normalization(B_mag))
            except:
                colors = plt.cm.plasma(normalization(B_mag))
        else:
            colors = 'black'

        if 'length' in kwargs and type(kwargs['length']) == float:
            l = kwargs['length']/B_mag.max()
        else:
            l = 1/B_mag.max()
        #finally, plotting the vector field
        axis.quiver(Xf, Yf, Zf, Bxf * map, Byf * map, Bzf * map, color=colors, length=l)
        if plot_charge:
            Xq,Yq,Zq,colorQ,sizes  = [],[],[],[],[]
            if 'charge_size' in kwargs:
                scale = kwargs['charge_size']
            else:
                scale = 100
            for charge in self.charges:
                Xq.append(charge.position[0]),Yq.append(charge.position[1]),Zq.append(charge.position[2])
                colorQ.append('red' if charge.charge>=0 else 'blue')
                sizes.append(np.abs(np.abs(charge.charge)*scale))
            axis.scatter(Xq,Yq,Zq,color=colorQ,s=sizes)
        plt.show()

    def esVisual2d(self, boundary_1=(-10, 10), boundary_2=(-10, 10), num=(10, 10), parallel_plane='xy', slice_plane=0,
                   **kwargs):
        ...