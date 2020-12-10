import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

class NbodyClass:
    def __init__(self, x, v, m=1, ngrid=10, dt=0.1, outdir='outputs/'): #WAIT I PROBABLY NEED vx, vy
        self.x = x.copy() #position vector.have x[0, :] = x, x[1,:] = y, x[2, :]=z etc.
        self.v = v.copy() #velocity vector. 
        self.m = m        #mass of a single particle
        self.dt = dt      #time step
        self.steps_taken = 0
        
        #defining the grid
        self.ngrid = ngrid
        self.grid = np.zeros((ngrid, ngrid))
        
        #calculate greens of the laplacian for a single particle
        self.calculate_greens()
        
        self.outdir = outdir
        
    def calculate_greens(self):
        dx = np.arange(self.ngrid)
        #making it wrap around. stuff in same column/row/depth will have 0
        #previous wraps around to -1, next goes to 1
        dx[self.ngrid//2:] = dx[self.ngrid//2:]-self.ngrid
#         print('dx is ', dx)
        #distances away from particle in x, y, z
        #particle lives at 0, 0, 0
        xmesh, ymesh, zmesh = np.meshgrid(dx,dx,dx)
#         mpl.imshow(xmesh[1, :, :], cmap='hot')
#         mpl.title('xmesh0')
#         mpl.show()
#         mpl.imshow(xmesh[5, :, :], cmap='hot')
#         mpl.show()
#         mpl.imshow(xmesh[9, :, :], cmap='hot')
#         mpl.show()
#         print(xmesh.shape)
#         mpl.imshow(ymesh[0, :, :], cmap='hot')
#         mpl.show()
#         mpl.imshow(zmesh[0, :, :], cmap='hot')
#         mpl.show()
        
        dr = np.sqrt(xmesh**2 + ymesh**2 + zmesh**2)
        mpl.imshow(dr[0, :, :], cmap='hot')
        mpl.title('dr')
        mpl.show()
#         mpl.imshow(dr[:, 0, :], cmap='hot')
#         mpl.show()
#         mpl.imshow(dr[:, :, 0], cmap='hot')
#         mpl.show()

        #point at 0,0,0 is 0 and will blow up
        dr[0, 0, 0] = 1
        #typically in physics define potential to be negative and go to zero infinitely far away. negative sign.
        self.pot = -1/(4*np.pi*dr)
#         print('potential: ', self.pot[0, :, :])
        mpl.imshow(self.pot[0, :, :], cmap='hot')
        mpl.title('potential')
        mpl.show()
        mpl.imshow(self.pot[:, 0, :], cmap='hot')
        mpl.title('potential')
        mpl.show()
        mpl.imshow(self.pot[:, :, 0], cmap='hot')
        mpl.title('potential')
        mpl.show()
        mpl.imshow(self.pot[:, :, 5], cmap='hot')
        mpl.title('potential')
        mpl.show()
    
    #convolve the density matrix with the potential
    def get_potential(self):
        #convolve the density matrix with the potential
        return 0
    
    #update the grid so we know how many particles per grid cell
    def update_grid(self):
        #sum ceil of positions as x/y grids.
        #toggle between period and nonperiodic boundary conditions
        #test without changing velocity and position
        return 0
        
    def take_step(self):
        #FOR A SINGLE PARTICLE
        #take the GRADIENT of the potential at its position
        #gives you the FORCE.
        #get the acceleration.
        #update the VELOCITY.
        #update the POSITION.
        #leapfrog somehow.
        
        #LEAP FROGGOOO
        #OK SO:
        #take x halfway into interval with currently estimated velocity
        #use THAT value of x to calculate the forces (so THAT'S when you'd call get potential I think. but would have to do for ALL of the particles
        #use THAT calculated force to get the velocity in the middle of the interval
        #use THAT velocity to update the x for real with a full timestep
        #then use the same force to update the velocity for real with a full timestep.
        
        #QUESTION: put ALL particles halfway out of position before calculating the potential? or just the one particle. probably all of the particles.
        self.x += self.dt*self.v
        return 0
        
    def plot_positions(self):
        fig=mpl.figure(figsize=(10,10))#Create 3D axes
        try: 
            ax=fig.add_subplot(111,projection="3d")
        except : ax=Axes3D(fig) 
        ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker=".", s=5)#,s=.02)
        ax.set_xlabel("x-coordinate",fontsize=14)
        ax.set_ylabel("y-coordinate",fontsize=14)
        ax.set_zlabel("z-coordinate",fontsize=14)
        ax.set_title("Particle Positions\n",fontsize=20)
        # ax.legend(loc="upper left",fontsize=14)
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])
        mpl.savefig(self.outdir+'nbodystep'+str(self.steps_taken)+'.png', dpi=1200)
    

    def run_nbody(self, iters=10):
        #start the first plot
        return 0
        