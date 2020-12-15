import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

class NbodyClass:
    #guard cells should be even. irrelevant for periodic BC
    def __init__(self, x, v, m=1, sgrid=10, dt=0.1, outdir='outputs/', periodic=False, guard=2, max_step=10**5, gradient=False, plot_name='nbodystep'):
        self.x = x.copy() #position vector.have x[0, :] = x, x[1,:] = y, x[2, :]=z etc.
        self.x_half = x.copy()
        self.v = v.copy() #velocity vector. 
        self.v_half = v.copy()
        self.m = m        #mass of a single particle
        self.dt = dt      #time step
        self.steps_taken = 0
        self.guard=guard
        self.plot_name = plot_name
        
        #defining the grid
        #ONLY WANT GUARD CELLS FOR NON PERIODIC
        if periodic: #offset x positions by guard cells to allow guard cells on zero side as well.
            self.sgrid = sgrid #the smaller grid of actual data
            self.ngrid = sgrid #the grid with gaurd cells
        else:
            self.x += guard//2 
            self.sgrid = sgrid #the smaller grid of actual data
            self.ngrid = sgrid + guard #the grid with gaurd cells
            
        self.grid = np.zeros((self.ngrid, self.ngrid, self.ngrid))
        self.update_grid()
        
        
        #deal with periodicity
        self.periodic = periodic
        
        #pre-define acceleration
        self.acc = np.zeros(x.shape)
        self.gradient = gradient
        
        #total energy, kinetic and potential
        self.ke = np.zeros( (3, max_step) )
        self.pe = np.zeros( (max_step) )
        
        #where to save output images
        self.outdir = outdir
        
        #calculate greens of the laplacian for a single particle
        #creates self.greens_pot
        self.greens_pot = 0
        self.calculate_greens()
        
        #initialize potential
        self.pot = np.zeros((self.ngrid, self.ngrid, self.ngrid))
                
    def calculate_greens(self, DEBUG=False):
        if self.periodic:
            num = self.ngrid
        else: #NOT periodic. will need to pad extra values of green's function
            num = 2*self.ngrid-1
            
        dx = np.arange(num)            
        #making it wrap around. stuff in same column/row/depth will have 0
        #previous wraps around to -1, next goes to 1
        dx[num//2:] = dx[num//2:]-num
            
        #distances away from particle in x, y, z
        #particle lives at 0, 0, 0
        xmesh, ymesh, zmesh = np.meshgrid(dx,dx,dx)
        dr = np.sqrt(xmesh**2 + ymesh**2 + zmesh**2)
        
        if DEBUG:
            print('dr x = 0 is \n ', dr[0, :, :])
            print('dr x = 1 is \n,', dr[1, :, :])
            mpl.imshow(dr[0, :, :], cmap='hot')
            mpl.title('dr x=0 2D slice')
            mpl.show()

        #point at 0,0,0 is 0 and will blow up
        #this is essentially softening potential, when it's that grid cell it's 1
#         dr[0, 0, 0] = 1
        
        #try softening a little more
        soft_idx = dr < 4
        dr[soft_idx] = 4
        
        if DEBUG:
            print('dr x = 0 is \n ', dr[0, :, :])
            print('dr x = 1 is \n,', dr[1, :, :])
            mpl.imshow(dr[0, :, :], cmap='hot')
            mpl.title('dr x=0 2D slice')
            mpl.show()

        self.greens_pot = 1/(4*np.pi*dr)
    
    #convolve the density matrix with the potential
    def calculate_potential(self, DEBUG=False):
        #FIRST UPDATE: self.grid (density matrix)
        #convolution in time domain equal to multiplication in frequency domain
        
        #probably need to know periodic or nonperiodic
        #nonperiodic is going to need to be zero padded
        
        if self.periodic: #periodic doesn't need padding
            fft_greens = np.fft.fftn(self.greens_pot)
            fft_density = np.fft.fftn(self.grid)

            self.pot = np.abs(np.fft.ifftn(fft_density * fft_greens))
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
            self.pot = np.abs(pot_padded[0:self.ngrid, 0:self.ngrid, 0:self.ngrid])
            
            if DEBUG:
                mpl.imshow(np.abs(pot_padded[0, :, :]), cmap='hot')
                mpl.title('PADDED final potential x=0 2D slice')
                mpl.show()

        if DEBUG:
            mpl.imshow(self.pot[0, :, :], cmap='hot')
            mpl.title('final potential x=0 2D slice')
            mpl.show()

            mpl.imshow(self.pot[self.ngrid//2, :, :], cmap='hot')
            mpl.title('final potential x=' + repr(self.ngrid//2) + ' 2D slice' )
            mpl.show()

            print('the particles are at ', np.nonzero(self.grid))        
        return 0
    
    def calculate_acceleration(self, DEBUG=False):
        #UPDATE THE GRADIENT FIRST. (and for the gradient, the density grid)
        #take the GRADIENT of the potential to get the force. divide by mass
        #use the half step of x to do the calculation
        num = self.x_half.shape[1]
        
        for ii in range(0, num):
            x_idx = self.x_half[:, ii].astype(int)
            x_idx_tup = tuple(x_idx)
            part_pot = np.roll(self.greens_pot, x_idx, (0, 1, 2))[0:self.ngrid, 0:self.ngrid, 0:self.ngrid]
            potential = self.pot - part_pot
            
            #update kinetic and potential energy for previous step
            self.pe[self.steps_taken] += potential[x_idx_tup]
            self.ke[:, self.steps_taken] = 1/2*self.m*(self.v[:, ii])**2
            
            if DEBUG:
                mpl.figure()
                mpl.imshow(self.greens_pot[0, :, :], cmap='hot')
                mpl.title('greens pot, x=0')
                mpl.colorbar()
                mpl.show()

                mpl.figure()
                mpl.imshow(part_pot[0, :, :], cmap='hot')
                mpl.title('part_pot, x=0' + repr( self.x_half[:, ii].astype(int)) )
                mpl.colorbar()
                mpl.show()

                mpl.figure()
                mpl.imshow(self.pot[0, :, :], cmap='hot')
                mpl.title('current potential of whole system, x=0')
                mpl.colorbar()
                mpl.show()

                mpl.figure()
                mpl.imshow(potential[0, :, :], cmap='hot')
                mpl.title('part_total, total potential particle sees, x=0')
                mpl.colorbar()
                mpl.show()
                    
            #GRADIENT: TWO possible methods.
            #1: np.gradient. cons: calculates for ENTIRE matrix. but our matrix
            #is DIFFERENT every time because the potential of the individual particle is gone.
            #2: upwind derivative. check velocity. use cell that you're moving towards...?
        
            #METHOD 1: np.gradient
            if self.gradient:
                dx, dy, dz = np.gradient(potential)
                try:
                    self.acc[0, ii] = dx[x_idx_tup]/self.m #divide by mass, which is just 1 by default. WILL NEED TO CHANGE.
                    self.acc[1, ii] = dy[x_idx_tup]/self.m
                    self.acc[2, ii] = dz[x_idx_tup]/self.m
                except Exception as e:
                    print(str(e))
                    print('bruh it broke wait why??')
                    print('x index is ', x_idx)
                    print('num is', num)
                    print('ii is', ii)
                    print('x half is ', self.x_half)
                    print('step number:  ', self.steps_taken)

                    assert 1==0

                if DEBUG:
                    mpl.figure()
                    mpl.imshow(dx[0, :, :], cmap='hot')
                    mpl.title('full gradient, x=0')
                    mpl.colorbar()
                    mpl.show()
                    print('Gradient method provided ', self.acc[:, ii])
            
            #METHOD 2: upwind derivative
            #check velocity to see which cell to differentiate.
            #will need guard cells for non-periodic.
            #will need to wrap around for periodic.
            
            else:
                try:
                    if self.periodic:
                        #IF PERIODIC:
                        #check velocity direction
                        #index with modulo
                        #subtract upwind one
                        #yeah... I'm sorry.
                        x_idx_mod = tuple(x_idx%self.ngrid)
                        x_idx_mod_plus = tuple((x_idx+1)%self.ngrid)
                        x_idx_mod_minus = tuple((x_idx-1)%self.ngrid)

                        if self.v[0, ii] < 0:
                            self.acc[0, ii] = potential[x_idx_mod] - potential[x_idx_mod_minus[0], x_idx_mod[1], x_idx_mod[2]]
                        elif self.v[0, ii] >= 0:
                            self.acc[0, ii] = potential[x_idx_mod_plus[0], x_idx_mod[1], x_idx_mod[2]] - potential[x_idx_mod]
                        else:
                            print('error periodic x force calc')

                        if self.v[1, ii] < 0:
                            self.acc[1, ii] = potential[x_idx_mod] - potential[x_idx_mod[0], x_idx_mod_minus[1], x_idx_mod[2]]
                        elif self.v[1, ii] >= 0:
                            self.acc[1, ii] = potential[x_idx_mod[0], x_idx_mod_plus[1], x_idx_mod[2]] - potential[x_idx_mod]
                        else:
                            print('error periodic y force calc')

                        if self.v[2, ii] < 0:
                            self.acc[2, ii] = potential[x_idx_mod] - potential[x_idx_mod[0], x_idx_mod[1], x_idx_mod_minus[2]]
                        elif self.v[2, ii] >= 0:
                            self.acc[2, ii] = potential[x_idx_mod[0], x_idx_mod[1], x_idx_mod_plus[2]] - potential[x_idx_mod]
                        else:
                            print('error periodic z force calc')

                        #then divide by mass to actually get acceleration
                        self.acc /= self.m

                    else:
                        #NOT periodic. need to check. is it towards the edge? if so, then subtract/add from zero.
                        if (self.v[0, ii] < 0) and (x_idx_tup[0] - 1 > 0):
                            self.acc[0, ii] = potential[x_idx_tup] - potential[x_idx_tup[0]-1, x_idx_tup[1], x_idx_tup[2]]
                        elif (self.v[0, ii] < 0):
                             self.acc[0, ii] = potential[x_idx_tup] - 0 #assume potential out of guard cells is zero
                        elif (self.v[0, ii] >= 0) and (x_idx_tup[0] + 1 < self.ngrid):
                            self.acc[0, ii] = potential[x_idx_tup[0]+1, x_idx_tup[1], x_idx_tup[2]] - potential[x_idx_tup]
                        elif (self.v[0, ii] >= 0):
                            self.acc[0, ii] = -1*potential[x_idx_tup] #assume potential out of guard cells is zero
                        else:
                            print('error nonperiodic x force calc')

                        if (self.v[1, ii] < 0) and (x_idx_tup[1] - 1 > 0):
                            self.acc[1, ii] = potential[x_idx_tup] - potential[x_idx_tup[0], x_idx_tup[1]-1, x_idx_tup[2]]
                        elif (self.v[1, ii] < 0):
                             self.acc[1, ii] = potential[x_idx_tup] - 0 #assume potential out of guard cells is zero
                        elif (self.v[1, ii] >= 0) and (x_idx_tup[1] + 1 < self.ngrid):
                            self.acc[1, ii] = potential[x_idx_tup[0], x_idx_tup[1]+1, x_idx_tup[2]] - potential[x_idx_tup]
                        elif (self.v[1, ii] >= 0):
                            self.acc[1, ii] = -1*potential[x_idx_tup] #assume potential out of guard cells is zero
                        else:
                            print('error nonperiodic y force calc')

                        if (self.v[2, ii] < 0) and (x_idx_tup[2] - 1 > 0):
                            self.acc[2, ii] = potential[x_idx_tup] - potential[x_idx_tup[0], x_idx_tup[1], x_idx_tup[2]-1]
                        elif (self.v[2, ii] < 0):
                             self.acc[2, ii] = potential[x_idx_tup] - 0 #assume potential out of guard cells is zero
                        elif (self.v[2, ii] >= 0) and (x_idx_tup[2] + 1 < self.ngrid):
                            self.acc[2, ii] = potential[x_idx_tup[0], x_idx_tup[1], x_idx_tup[2]+1] - potential[x_idx_tup]
                        elif (self.v[2, ii] >= 0):
                            self.acc[2, ii] = -1*potential[x_idx_tup] #assume potential out of guard cells is zero
                        else:
                            print('error nonperiodic z force calc')
                except Exception as e:
                    print(str(e))
                    print('periodic: ', str(self.periodic))
                    print('particle number ', ii, ' step number ', self.steps_taken)
                    print('current position ', self.x[:, ii])
                
                if DEBUG:
                    print('upwind/downwind method provided ', self.acc[:, ii])

                
#                 sign = np.sign(self.v[:, ii])
#                 sign[sign==0] = 1 #no sign, just take from cell + 1
#                 print('velocity is', self.v[:, ii])
#                 print('sign is ', sign)
#                 #if sign is NEGATIVE: backwards step. p(x) - p(x - dx)
#                 #if sign is POSITIVE: FORWARDS step. p(x + dx) - p(x)
#                 self.acc[0, ii] = -1*sign[0]*potential[x_idx%self.ngrid] + sign[0]*potential[                     (x_idx[0]+sign[0])%self.ngrid, x_idx[1]%self.ngrid, x_idx[2]%self.ngrid]
#                 self.acc[1, ii] = -1*sign[1]*potential[x_idx%self.ngrid] + sign[1]*potential[x_idx[0], x_idx[1]+sign[1], x_idx[2]]
#                 self.acc[2, ii] = -1*sign[2]*potential[x_idx%self.ngrid] + sign[2]*potential[x_idx[0], x_idx[1], x_idx[2]+sign[2]]
                    
                
            
        #FOR EACH PARTICLE
            #need to get the single particles effect.
            #really just shifted version of green's potential
            #shifted so that the center is AT the particle.
            #then subtracted off.
            #probably use some mods or soemthing? could multiply in frequency space.
                #but that would be less efficient. just shift with matrix magic or sth.
            #will be bigger/smaller for periodic/nonperiodic so watch out for that.   
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
        return 0

    def update_grid_half(self):
        #COPIED CODE FROM UPDATE_GRID but using self.x_half now.
        #yeah... I realize I could just PASS in the data... or use toggle...
        self.grid[:, :, :] = 0 #CLEAR GRID
        
        #determine which particles are still inside grid
        #CUTOFF VERSION
        idx = (self.x_half>0) & (self.x_half<self.ngrid)
        idx_inside = idx[0, :] & idx[1, :] & idx[2, :]
        x_inside = self.x_half[:, idx_inside]
        
        #turn the positions into grid indices
        x_grid_pos = tuple(map(tuple, x_inside.astype(int))) #MADE INTO INDICES
        np.add.at(self.grid, x_grid_pos, 1) #UPDATE GRID
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
        
    def take_step(self, DEBUG=False):
        #QUESTION: put ALL particles halfway out of position before calculating the potential? or just the one particle. probably all of the particles

        #TO DO:
        #take X halfway into interval with currently estimated velocity (HALF timestep)
        # use THAT value of the x to calculated the...
            #potential
            #gradient of potential: force
            #new velocity in the MIDDLE of the interval (with HALF timestep)
        #update X: use that velocity in the middle of the interval, full timestep
        #update V: use FULL timestep now
        
        #get halfway done x values. update grid and potential based on them.
        
        if self.periodic:
            self.x_half = (self.x + 1/2*self.dt*self.v)%self.ngrid #keep it inside the size
        else:
            x_half_tmp = self.x + 1/2*self.dt*self.v
            idx = (x_half_tmp >= 0) & (x_half_tmp < self.ngrid) #get rid of lost particles
            idx_inside = idx[0, :] & idx[1, :] & idx[2, :]
            
            #cut it from all vectors with particles
            self.x_half = x_half_tmp[:, idx_inside]
            self.x = self.x[:, idx_inside]
            self.v_half = self.v_half[:, idx_inside]
            self.v = self.v[:, idx_inside]
            self.acc = self.acc[:, idx_inside]
            
        self.update_grid_half() #updates grid to values of half x
        self.calculate_potential(DEBUG=DEBUG)
        
        #calculate the acceleration based on halfway x positions and potential
        self.calculate_acceleration(DEBUG=DEBUG)
        
        #calculate new halfway velocity in the middle of the interval
        self.v_half = self.v + 1/2*self.dt*self.acc
        
        #use that to fully update update x and v.
        if self.periodic:
            self.x = (self.x + self.dt*self.v_half)%self.ngrid #keep it inside the size
        else:
            x_tmp = self.x + self.dt*self.v_half
            idx = (x_tmp >= 0) & (x_tmp < self.ngrid) #get rid of lost particles
            idx_inside = idx[0, :] & idx[1, :] & idx[2, :]
            
            #cut it from all vectors with particles
            self.x_half = self.x_half[:, idx_inside]
            self.x = x_tmp[:, idx_inside]
            self.v_half = self.v_half[:, idx_inside]
            self.v = self.v[:, idx_inside]
            self.acc = self.acc[:, idx_inside]
        
        self.v = self.v + self.dt*self.acc
        
        #update the grid with the new legit x positions
        self.update_grid()
        
        self.steps_taken += 1
        return 0
    
    
    
    
    def take_n_steps(self, n, plots=False, plot_every=10, DEBUG=False):
        for ii in range(0, n):
            self.take_step(DEBUG=DEBUG)
            if DEBUG:
                print('steps done: ', self.steps_taken)
            
            #plot positions every.... 10 steps
            if plots and (ii%plot_every == 0):
                self.plot_positions()
        return self.steps_taken
        
    
    #the x position has guard cells that shift its position.
    #to return original position need to subtract off the added guard cells
    #but ONLY for NONPERIODIC case
    def get_x(self):
        if self.periodic:
            return self.x
        return self.x - self.guard//2 
    
    def plot_positions(self):
        fig=mpl.figure(figsize=(10,10))#Create 3D axes
        try: 
            ax=fig.add_subplot(111,projection="3d")
        except : ax=Axes3D(fig) 
        ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker="*", s=0.02)#,s=.02)
#         ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker=".", s=0.9)#,s=.02)
        ax.set_xlabel("x-coordinate",fontsize=14)
        ax.set_ylabel("y-coordinate",fontsize=14)
        ax.set_zlabel("z-coordinate",fontsize=14)
        ax.set_title("Particle Positions step" + str(self.steps_taken),fontsize=20)
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
        
        #make a consistent 2 digit naming to satisfy the haters aka ffpmeg
        if self.steps_taken < 10: 
            stepstr = '0' + str(self.steps_taken)
        else:
            stepstr = str(self.steps_taken)
#         mpl.savefig(self.outdir+'nbodystep'+str(self.steps_taken)+'.png', dpi=1200) #way too high quality
#         mpl.savefig(self.outdir+'nbodystep'+str(self.steps_taken)+'test1.jpg', dpi=400) #definitely ok
        mpl.savefig(self.outdir + self.plot_name + stepstr +'.jpg', dpi=250) #maybe ok?
#         mpl.savefig(self.outdir+'nbodystep'+str(self.steps_taken)+'3.jpg', dpi=100) #too low

        mpl.close(fig) #don't want to display it, just save it.
    
    def plot_energy(self):
        mpl.figure()
        mpl.plot(range(0, self.steps_taken), self.pe[:self.steps_taken])
        mpl.plot(range(0, self.steps_taken), self.ke[0, :self.steps_taken])
        mpl.plot(range(0, self.steps_taken), self.ke[1, :self.steps_taken])
        mpl.plot(range(0, self.steps_taken), self.ke[2, :self.steps_taken])
        mpl.plot(range(0, self.steps_taken), self.pe[:self.steps_taken] + np.sum(self.ke, axis=0)[:self.steps_taken])
        mpl.legend(['Pot', 'keX', 'keY', 'keZ', 'tot e'])
        mpl.title('energy')
        mpl.show()