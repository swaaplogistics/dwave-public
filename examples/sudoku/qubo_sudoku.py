############################################################
#							   #
#   D-Wave QUBO formulation sudoku solver		   #
#							   #
#   July 2022 by Emilio Pomares (epomares@swaap.us)	   #
#							   #
############################################################

# Imports

from sudoku import Sudoku as s
import numpy as np
from dimod import BinaryQuadraticModel
from dimod import SimulatedAnnealingSampler
from dwave.system.composites import EmbeddingComposite
from dimod import Binary
from dimod import BINARY

from dwave.system import DWaveSampler
#sampler = EmbeddingComposite(DWaveSampler(token='YOUR API KEY HERE'))
sampler = SimulatedAnnealingSampler() # Select Simulated Annealer for testing

# Define sizes

size = 3
maxRange = size*size

# Generate a puzzle and visualize it

puzzle = s(size).difficulty(0.8)
puzzle.show()

# See board data

puzzle.board

# Initialize linear and quadratic coeffs dictionaries

linear = {}
quadratic = {}

# QUBO coefficients building functions

def add_linear(index, val):
    if not index in linear:
        linear[index] = val
    else:
        linear[index] += val
        
def add_quad(indexA, indexB, val):
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
    return i*maxRange*maxRange + j*maxRange + k

def add_eq_constraint_coeffs(i, j, k):
    for sk in range(0, maxRange):
        if k == sk:
            add_linear(linear_index(i,j,sk), -1)
        else:
            add_linear(linear_index(i,j,sk), 1)
    return

def add_just_one_constraints_coeffs():
    for i in range(0,maxRange):
        for j in range(0,maxRange):
            for k in range(0,maxRange):
                for w in range(0,maxRange):
                    add_quad(linear_index(i,j,k), linear_index(i,j,w), 1)
                    add_quad(linear_index(i,k,j), linear_index(i,w,j), 1)
                    add_quad(linear_index(k,i,j), linear_index(w,i,j), 1)
                add_linear(linear_index(i,j,k), -1)
                add_linear(linear_index(i,k,j), -1)
                add_linear(linear_index(k,i,j), -1)
    return

# Create existing numbers in board constraints

for i in range(0,maxRange):
    for j in range(0,maxRange):
        if puzzle.board[i][j]:
            add_eq_constraint_coeffs(i,j,puzzle.board[i][j]-1)
            
add_just_one_constraints_coeffs()

# Construct the bqm from the linear and quadratic coefficients

bqm = BinaryQuadraticModel(linear, quadratic, 0.0, BINARY)

# Be sure to perform 'enough' reads
num_reads = 500
if isinstance(sampler, SimulatedAnnealingSampler):
    num_reads = 50

sample_set = sampler.sample(bqm,num_reads=num_reads,label="QUBO Sudoku")
best = sample_set.first.sample
flattened_solution = [best[key] for key in sample_set.first.sample]

# Utility to reconstruct one-hot encoded number from solution

def get_board_value(i, j, output):
    for k in range(0,maxRange):
        index = linear_index(i, j, k)
        if(output[index]) > 0.0:
            return k+1
    return None

def draw_row_separator():
    tempstr = ""
    for w in range(0,size):
        tempstr += "+"
        for k in range(0,2*size+1):
            tempstr += "-"
    tempstr += "+"
    print(tempstr)

def draw_board(output):
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

