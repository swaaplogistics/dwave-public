#############################################################
#							                                #
#   D-Wave BQM/QUBO formulation sudoku solver	            #
#							                                #
#   July 2022 by Emilio Pomares (epomares@swaap.us)	        #
#							                                #
#############################################################

# Imports

from sudoku import Sudoku as s
import numpy as np
from dimod import BinaryQuadraticModel
from dimod import SimulatedAnnealingSampler
from dwave.system.composites import EmbeddingComposite
from dimod import Binary
from dimod import BINARY

from dwave.system import DWaveSampler
from dwave.system import LeapHybridSampler

# select a sampler by commenting/uncommenting:

#sampler = EmbeddingComposite(DWaveSampler(token='Your API Key here'))
sampler = LeapHybridSampler(token='Your API Key here')
#sampler = SimulatedAnnealingSampler()

# Define sizes

size = 3
maxRange = size*size

# Generate a puzzle and visualize it

puzzle = s(size).difficulty(0.8)
print(f" A {maxRange}x{maxRange} sudoku challenge: ")
puzzle.show()

# Use built-in solver to obtain a solution...

print(f" Let's use py-sudoku's built-in solver to see one possible solution:")
puzzle.solve().show()

# Show board data encodings
print("The board encoding: ")
print(puzzle.board)

# Let's encode the number in each cell as a one-hot encoded vector of length `maxRange`, 
# thus, the sudoku board becomes a cube of maxRange x maxRange x maxRange binary variables

# Initialize linear and quadratic coeffs dictionaries

linear = {}
quadratic = {}

# We need to add three families of constraints to solve a sudoku: 1) numbers present in challenge must
# be preserved, 2) numbers cannot repeat accross y-columns, z-columns and rows, 3) numbers cannot
# repeat across size x size cells

# QUBO coefficients building functions

def map_linear_to_cell(k, cell_size=size):
    '''
    Maps a cell's linear index in the range [0,size*size-1]
    to a (i, j) coordinate in the range [0, size-1], [0, size-1]
    '''
    return (k%cell_size, k//cell_size)

def add_linear(index, val):
    '''
    Adds influence to linear bias
    '''
    if not index in linear:
        linear[index] = val
    else:
        linear[index] += val
        
def add_quad(indexA, indexB, val):
    '''
    Adds coupling strength influence
    
    This function is repeated indices safe
    '''
    if indexA == indexB:
        return
    key = (indexA, indexB)
    if indexB > indexA:
        key = (indexB, indexA)
    if not key in quadratic:
        quadratic[key] = val
    else:
        quadratic[key] += val
        
def linear_index(i, j, k):
    '''
    Transforms from (i, j, k) coordinates in the range
    ([0, size*size-1], [0, size*size-1], [0, size*size-1])
    to a linear index in the range [0, (size*size)^3-1]
    '''
    return i*maxRange*maxRange + j*maxRange + k

def add_eq_constraint_coeffs(i, j, k):
    '''
    Helper function for add_existing_numbers_constraints
    '''
    for sk in range(0, maxRange):
        if k == sk:
            add_linear(linear_index(i,j,sk), -2) # should be -1, 1, but a little more strength...
        else:
            add_linear(linear_index(i,j,sk), 2)
    return

def add_existing_numbers_constraints():
    '''
    Adds influences related to numbers-in-board equality constraints
    to the QUBO matrix
    '''
    for i in range(0,maxRange):
        for j in range(0,maxRange):
            if puzzle.board[i][j]:
                add_eq_constraint_coeffs(i,j,puzzle.board[i][j]-1)

def add_just_one_constraints_coeffs():
    '''
    Adds influences related to non-repeating-numbers in rows,columns equality 
    constraints to the QUBO matrix
    '''
    for i in range(0,maxRange):
        for j in range(0,maxRange):
            for k in range(0,maxRange):
                for w in range(0,maxRange):
                    # add_quad will take care of repeated indices
                    add_quad(linear_index(i,j,k), linear_index(i,j,w), 2) 
                    add_quad(linear_index(i,k,j), linear_index(i,w,j), 2)
                    add_quad(linear_index(k,i,j), linear_index(w,i,j), 2)
                add_linear(linear_index(i,j,k), -1)
                add_linear(linear_index(i,k,j), -1)
                add_linear(linear_index(k,i,j), -1)
    return

def add_cell_constraints_coeffs():
    '''
    Adds influences related to the non-repeating-numbers-in-cell constraints
    to the QUBO matrix
    '''
    for i_cell in range(0, size):
        for j_cell in range(0, size):
            for depth in range(0, maxRange):
                for k in range(0, maxRange):
                    ik,jk=map_linear_to_cell(k)
                    for w in range(0, maxRange):
                        iw,jw=map_linear_to_cell(w)
                        # add_quad will take care of repeated indices
                        add_quad(linear_index(i_cell*size+ik,j_cell*size+jk,depth), linear_index(i_cell*size+iw,j_cell*size+jw,depth), 2) 
                    add_linear(linear_index(i_cell*size+ik,j_cell*size+jk,depth), -1)

# Create existing-numbers-in-board constraints

add_existing_numbers_constraints()

# Create axis just-one constraints

add_just_one_constraints_coeffs()

# Create cell just-one constraints

add_cell_constraints_coeffs()

# Construct the bqm from the linear and quadratic coefficients

bqm = BinaryQuadraticModel(linear, quadratic, 0.0, BINARY)

# Check number of variables

print(f"We are working with these many binary variables: {len(bqm.variables)}")

# make sure to select 'enough' samples

num_reads = 1500
if isinstance(sampler, SimulatedAnnealingSampler):
    num_reads = 150
if isinstance(sampler, LeapHybridSampler):
    sample_set = sampler.sample(bqm,label="QUBO Sudoku (Hybrid)")
elif isinstance(sampler, SimulatedAnnealingSampler):
    sample_set = sampler.sample(bqm,num_reads=num_reads,label="QUBO Sudoku (Simulated)")
    print(num_reads)
else:
    sample_set = sampler.sample(bqm,num_reads=num_reads,label="QUBO Sudoku (QPU)")
    print(num_reads)

# Select best sample

best = sample_set.first.sample

# Recover solution

flattened_solution = [best[key] for key in sample_set.first.sample]

# Some helper functions to visualize the solution

def get_board_value(i, j, output):
    '''
    Reconstructs one-hot encoded number from the flattened solution
    '''
    for k in range(0,maxRange):
        index = linear_index(i, j, k)
        if(output[index]) > 0.0:
            return k+1
    return None

def draw_row_separator():
    '''
    Draws a board row separator as ASCII characters
    '''
    tempstr = ""
    for w in range(0,size):
        tempstr += "+"
        for k in range(0,2*size+1):
            tempstr += "-"
    tempstr += "+"
    print(tempstr)

def draw_board(output):
    '''
    Draws the sudoku board as ASCII characters
    '''
    for i in range(0,maxRange):
        if i%size == 0:
            draw_row_separator()
        tempstr = ""
        for j in range(0,maxRange):
            if j%size == 0:
                tempstr += "| "
            tempstr += (str(get_board_value(i,j,output)) + " ")
        tempstr += "|"
        print(tempstr)
    tempstr = ""
    draw_row_separator()

# Draw the solution board

draw_board(flattened_solution)

# Original board

puzzle.show()

