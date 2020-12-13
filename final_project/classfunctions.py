import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

class NbodyClass:
    def __init__(self, x, v, m=1, ngrid=10, dt=0.1, outdir='outputs/', periodic=False): #WAIT I PROBABLY NEED vx, vy
        self.x = x.copy() #position vector.have x[0, :] = x, x[1,:] = y, x[2, :]=z etc.
        self.v = v.copy() #velocity vector. 
        self.m = m        #mass of a single particle
        self.dt = dt      #time step
        self.steps_taken = 0
        
        #deal with periodicity
        self.periodic = periodic
        
        #where to save output images
        self.outdir = outdir
        
        #defining the grid
        self.ngrid = ngrid
        self.grid = np.zeros((ngrid, ngrid, ngrid))
        self.update_grid()
        
        #calculate greens of the laplacian for a single particle
        #creates self.greens_pot
        self.greens_pot = 0
        self.calculate_greens()
        
        #initialize potential
        self.pot = np.zeros((ngrid, ngrid, ngrid))
                
    def calculate_greens(self):
        if self.periodic:
            num = self.ngrid
#             dx = np.arange(self.ngrid)
            #making it wrap around. stuff in same column/row/depth will have 0
            #previous wraps around to -1, next goes to 1
#             dx[self.ngrid//2:] = dx[self.ngrid//2:]-self.ngrid
    #         print('dx is ', dx)
            #distances away from particle in x, y, z
            #particle lives at 0, 0, 0
        else: #NOT periodic. will need to pad extra values of green's function
            num = 2*self.ngrid-1
        dx = np.arange(num)
        dx[num//2:] = dx[num//2:]-num
            
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
#         mpl.imshow(dr[0, :, :], cmap='hot')
#         mpl.title('dr')
#         mpl.show()
#         mpl.imshow(dr[:, 0, :], cmap='hot')
#         mpl.show()
#         mpl.imshow(dr[:, :, 0], cmap='hot')
#         mpl.show()

        #point at 0,0,0 is 0 and will blow up
        dr[0, 0, 0] = 1
        #typically in physics define potential to be negative and go to zero infinitely far away. negative sign.
        self.greens_pot = -1/(4*np.pi*dr)
#         print('potential: ', self.greens_pot[0, :, :])
#         mpl.imshow(self.greens_pot[0, :, :], cmap='hot')
#         mpl.title('potential')
#         mpl.show()
#         mpl.imshow(self.greens_pot[:, 0, :], cmap='hot')
#         mpl.title('potential')
#         mpl.show()
#         mpl.imshow(self.greens_pot[:, :, 0], cmap='hot')
#         mpl.title('potential')
#         mpl.show()
#         mpl.imshow(self.greens_pot[:, :, 5], cmap='hot')
#         mpl.title('potential')
#         mpl.show()
    
    #convolve the density matrix with the potential
    def get_potential(self, DEBUG=False):
        #convolution in time domain equal to multiplication in frequency domain
        
        #probably need to know periodic or nonperiodic
        #nonperiodic is going to need to be zero padded
        
        if self.periodic: #periodic doesn't need padding
            fft_greens = np.fft.fftn(self.greens_pot)
            fft_density = np.fft.fftn(self.grid)

            self.pot = np.fft.ifftn(fft_density * fft_greens)
        else: #not periodic, need padding
            N = 2*self.ngrid - 1 #everything is square
    
            # DO NOT PAD GREENS POTENTIAL WITH ZEROS
            #WHY? because it's NOT zero off the grid. dies off slowly.
            #Instead in creation it has already been calculated to size N.

            #padding density: pad with zeros, assume particles are gone off grid
            #TO DO: keep BUFFER CELLS around the edge and pad with THOSE, and THEN zeros.
            density_padded = np.zeros( (N, N, N) )
            density_padded[0:self.ngrid, 0:self.ngrid, 0:self.ngrid] = self.grid
            
            fft_greens = np.fft.fftn(self.greens_pot)
            fft_density = np.fft.fftn(density_padded)

            pot_padded = np.fft.ifftn(fft_density * fft_greens)
            self.pot = pot_padded[0:self.ngrid, 0:self.ngrid, 0:self.ngrid]
            
            if DEBUG:
#                 mpl.imshow(self.greens_pot[0, :, :], cmap='hot')
#                 mpl.title('greens potential padded x=0 2D slice')
#                 mpl.show()
                
#                 mpl.imshow(density_padded[0, :, :], cmap='hot')
#                 mpl.title('density padded x=0 2D slice')
#                 mpl.show()
                
                mpl.imshow(np.abs(pot_padded[0, :, :]), cmap='hot')
                mpl.title('PADDED final potential x=0 2D slice')
                mpl.show()

        if DEBUG:
            mpl.imshow(np.abs(self.pot[0, :, :]), cmap='hot')
            mpl.title('final potential x=0 2D slice')
            mpl.show()

            mpl.imshow(np.abs(self.pot[self.ngrid//2, :, :]), cmap='hot')
            mpl.title('final potential x=' + repr(self.ngrid//2) + ' 2D slice' )
            mpl.show()

            print('the particles are at ', np.nonzero(self.grid))
        
        #         mpl.imshow(np.abs(fft_greens[0, :, :]), cmap='hot')
#         mpl.title('fft of potential')
#         mpl.show()
        
#         mpl.imshow(self.grid[0, :, :], cmap='hot')
#         mpl.title('density')
#         mpl.show()
        
#         mpl.imshow(np.abs(fft_density[0, :, :]), cmap='hot')
#         mpl.title('fft of density')
#         mpl.show()
        
#         print('greens', self.greens_pot)
#         print(fft_greens)
#         print('density ', self.grid)
#         print(fft_density)
        
        
        #convolve the density matrix with the potential
        return 0
    
    #update the grid so we know how many particles per grid cell
    def update_grid(self):
        #sum ceil of positions as x/y grids.
        #toggle between period and nonperiodic boundary conditions
        #test without changing velocity and position
        self.grid[:, :, :] = 0 #CLEAR GRID
        
        #determine which particles are still inside grid
        #CUTOFF VERSION
        idx = (self.x>0) & (self.x<self.ngrid)
        idx_inside = idx[0, :] & idx[1, :] & idx[2, :]
        x_inside = self.x[:, idx_inside]
        
        #turn the positions into grid indices
        x_grid_pos = tuple(map(tuple, x_inside.astype(int))) #MADE INTO INDICES
        np.add.at(self.grid, x_grid_pos, 1) #UPDATE GRID
#         print(self.x)
#         print(x_grid_pos)
#         print(self.grid)
        return 0

    def plot_density_heatmap(self):
        mpl.imshow(self.grid[0, :, :], cmap='hot')
        mpl.title('grid, x=0')
        mpl.show()
        
        mpl.imshow(self.grid[:, 0, :], cmap='hot')
        mpl.title('grid, y=0')
        mpl.show()
        
        mpl.imshow(self.grid[:, :, 0], cmap='hot')
        mpl.title('grid, z=0')
        mpl.show()
        
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
        self.steps_taken += 1
        
    def plot_positions(self):
        fig=mpl.figure(figsize=(10,10))#Create 3D axes
        try: 
            ax=fig.add_subplot(111,projection="3d")
        except : ax=Axes3D(fig) 
        ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker="*", s=0.9)#,s=.02)
#         ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker=".", s=0.9)#,s=.02)
        ax.set_xlabel("x-coordinate",fontsize=14)
        ax.set_ylabel("y-coordinate",fontsize=14)
        ax.set_zlabel("z-coordinate",fontsize=14)
        ax.set_title("Particle Positions\n",fontsize=20)
        # ax.legend(loc="upper left",fontsize=14)
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])
        
        #put invisible points at the corners to hold the view steady
        ax.scatter(0, 0, 0, alpha=0)
        ax.scatter(0, self.ngrid, 0, alpha=0)
        ax.scatter(0, 0, self.ngrid, alpha=0)
        ax.scatter(0, self.ngrid, self.ngrid, alpha=0)
        ax.scatter(self.ngrid, 0, 0, alpha=0)
        ax.scatter(self.ngrid, self.ngrid, 0, alpha=0)
        ax.scatter(self.ngrid, 0, self.ngrid, alpha=0)
        ax.scatter(self.ngrid, self.ngrid, self.ngrid, alpha=0)

        mpl.savefig(self.outdir+'nbodystep'+str(self.steps_taken)+'.png', dpi=1200)
    

    def run_nbody(self, iters=10):
        #start the first plot
        return 0
        