#############################################################
#							                                #
#   D-Wave CQM formulation sudoku solver		            #
#							                                #
#   July 2022 by Emilio Pomares (epomares@swaap.us)	        #
#							                                #
#############################################################

# Imports

from sudoku import Sudoku as s
import numpy as np
from dimod import SimulatedAnnealingSampler
from dimod import ConstrainedQuadraticModel
from dwave.system.composites import EmbeddingComposite
from dimod import Binary

from dwave.system import LeapHybridCQMSampler
from dwave.system import DWaveSampler

# We need a sampler that can work with CQM models:

sampler = LeapHybridCQMSampler(token='Your API Key here')

# Some helpers functions...

def get_board_value(i, j, output):
    '''
    Utility to reconstruct one-hot encoded number from solution
    '''
    for k in range(1,maxRange+1):
        if(output[f'SEL_{i}_{j}_{k}']) > 0.0:
            return k
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
    for i in range(1,maxRange+1):
        if (i-1)%size == 0:
            draw_row_separator()
        tempstr = ""
        for j in range(1,maxRange+1):
            if (j-1)%size == 0:
                tempstr += "| "
            tempstr += (str(get_board_value(i,j,output)) + " ")
        tempstr += "|"
        print(tempstr)
    tempstr = ""
    draw_row_separator()

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

# Initialize cqm

cqm = ConstrainedQuadraticModel()

# Create maxRange x maxRange x maxRange i_j_k binary variables

selected = [[[Binary(f'SEL_{i}_{j}_{k}') for k in range(1, maxRange+1)] 
             for j in range(1, maxRange+1)] for i in range(1, maxRange+1)]

# We need to add three families of constraints to solve a sudoku: 1) numbers present in challenge must
# be preserved, 2) numbers cannot repeat accross y-columns, z-columns and rows, 3) numbers cannot
# repeat across size x size cells

# Create existing-numbers-in-board constraints

for i in range(0,maxRange):
    for j in range(0,maxRange):
        if puzzle.board[i][j]:
            sk = puzzle.board[i][j]-1
            for k in range(0,maxRange):
                if k == sk:
                    cqm.add_constraint(selected[i][j][k] == 1, 
                        label = f'existing_number_{i+1}_{j+1}_{k+1}')
                else:
                    cqm.add_constraint(selected[i][j][k] == 0, 
                        label = f'existing_number_{i+1}_{j+1}_{k+1}')

# Create axis just-one constraints

for i in range(0,maxRange):
    for j in range(0,maxRange):
        sum_x = []
        sum_y = []
        sum_z = []
        for k in range(0,maxRange):
            sum_x.append(selected[i][j][k])
            sum_y.append(selected[j][k][i])
            sum_z.append(selected[k][i][j])
        cqm.add_constraint(sum(sum_x) == 1, 
            label = f'only_one_axis_X_{i+1}_{j+1}')
        cqm.add_constraint(sum(sum_y) == 1, 
            label = f'only_one_axis_Y_{i+1}_{j+1}')
        cqm.add_constraint(sum(sum_z) == 1, 
            label = f'only_one_axis_Z_{i+1}_{j+1}')


# Create cell just-one constraints

for i_cell in range(0,size):
    for j_cell in range(0,size):
        for k in range(0,maxRange):
            sum_cell = []
            for i in range(0, size):
                for j in range(0, size):   
                    sum_cell.append(selected[i_cell*size+i][j_cell*size+j][k])
            cqm.add_constraint(sum(sum_cell) == 1, label = f'only_one_cell_Z_{i_cell}_{j_cell}_{k}')

# Check number of variables

print(f"We are working with these many binary variables: {len(cqm.variables)}")

# Sample solutions

sampleset = sampler.sample_cqm(cqm,time_limit=25,label="CQM Sudoku") 
feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
n_solutions = len(feasible_sampleset)
if n_solutions == 0:
    print("No solutions found, try increasing time_limit!")
    assert 0==1
else:
    print("These many feasible solutions found: " + str(len(feasible_sampleset)))

# Select best sample and build solution dictionary

best = feasible_sampleset.first
best.sample.items()
output = {}
for key, value in best.sample.items():
    output[key]=value

# Draw the solution board

draw_board(output)

# Compare with input

puzzle.show()