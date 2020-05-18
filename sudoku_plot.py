#################################
# Ayden Ellsmere
# Sudoku CSP Solver - With Plots
#################################

import numpy as np
import copy
import matplotlib.pyplot as plt

# Sudoku Grid 9x9
GRIDSIZE = 9

### Globals ###
# Keeps track of number of variable assignments for reporting
global callcounter

# Keeps track of number backtracks
global backtrackcounter

# invalidmatrix: used in forward checking - full global 9x9x9 matrix where
# invalidmatrix[i][j][k] is 0 if the cell on the (i-1)th row and (j-1)th column can possibly be digit (k-1)
# e.g. invalidmatrix[2][3][7] = 0 implies that the cell at row 1, column 2 can potentially be 7
# invalidmatrix[i][j][k] is 1 if instead that digit is not valid to choose
# invalidmatrix[i][j][k] is 2 if that digit is not valid because of forward checking done by another cell
# Distinguish between above cases 1/2 so it would be possible to undo effects of foward checking without losing invalid info from other sources
global invalidmatrix

# Maximum iterations allowed before termination with failure
global maxiter
# Flag to to ensure initial setup of foward checking matrix only run once
global firsttimeflag
# Outputted solution - will be a 9x9 matrix
global solution

# Defaults
callcounter = 0
backtrackcounter = 0
maxiter = 10000
firsttimeflag = 1
invalidmatrix = np.zeros((GRIDSIZE, GRIDSIZE, GRIDSIZE))
solution = []

### findnextzero Helper Fucntion - Basic Version ###
# Function to locate next unassigned cell (with value zero) to try and fill
# Used by the basic Sudoku solver and forward checking solvers
# Start variable used to indicate next row of possible zero (unfilled cell) to save computation
# Since in the basic solver and version with forward checking are going row-by-row
# Puzzle is the Sudoku grid
def findnextzero(puzzle, start):
	# return [-1,-1] if no zeros found - indicates puzzle is solved
	result = [-1,-1]
	# Go through grid looking for value 0 in some cell, then break and return the location of that cell
	breakflag = 0
	for r in range(GRIDSIZE-start):
		for c in range(GRIDSIZE):
			if puzzle[r+start][c] == 0:
				result = [r+start,c]
				breakflag = 1
				break
		if breakflag == 1:
			break
	return result

### findnextzero - Improved Version with heuristics ###
# Version of findnextzero used by advanced Sudoku solver algorithm using most contrained variable and most constraining variable CSP heuristics
# Note: only annotating new info, refer to the more basic version for shared elements of algorithm
def findnextzeroheuristics(puzzle):
	# no more start variable since choices not based off row number now
	result = [-1,-1]
	# keep track of most retrictions for most constrained variable heuristic
	mostrestrictions = 0
	for r in range(GRIDSIZE):
		for c in range(GRIDSIZE):
			restrictions = 0
			# found a zero
			if puzzle[r][c] == 0:
				# go through this choice's invalid vector (9 elements) to look for non-zero values which indicates the choice of i-1 is invalid
				for i in range(GRIDSIZE):
					if invalidmatrix[r][c][i] != 0:
						restrictions += 1
				# if this choice has more restrictions (more non-zeros in invalid vector) then choose it over previous choice
				if restrictions > mostrestrictions:
					mostrestrictions = restrictions
					result = [r,c]
				# if there is a tie, use most constraining variable heuristic
				elif restrictions == mostrestrictions:
					# count the number of unassigned (0) cells on the same row/column as current choice and the previous best choice
					# which ever has more unassigned cells sharing its row/column (meaning it will likely restrict more choices if picked) will be chosen
					testr = 0
					mostr = 0
					for v in range(GRIDSIZE):
						if puzzle[result[0]][v] == 0 or puzzle[v][result[1]] == 0:
							mostr += 1
						if puzzle[r][v] == 0 or puzzle[v][c] == 0:
							testr += 1
					if(testr >= mostr):
						result = [r,c]
	return result

### findnextvalid Helper Function - Basic Version ###
# Function to find the lowest valid choice of digit for the given cell/variable
# This version is for the basic solver
# Puzzle is the Sudoku grid, pointtocheck is the cell we are trying to find a value (digit) assignment for
# invalid is the invalid vector for that cell - i.e. which digits are valid to assign to it
def findnextvalid(puzzle, pointtocheck, invalid):
	# assignment to choose
	assignment = 0
	# temp holder for value we want to test
	test = 0
	# flag for skipping further checking (e.g. moving on to checking columns) if value choice already found to be invalid
	badaflag = 0
	# try all possible values (1-9), if there are none then function will return 0
	while test <= 8:
		test += 1
		badaflag = 0
		# if this value was already marked invalid
		if invalid[test-1] == 1:
			badaflag = 1

		# check cells in the same row as pointtocheck, if there is already another cell sharing the value then this choice is invalid
		if badaflag == 0:
			for c in range(GRIDSIZE):
				if puzzle[pointtocheck[0]][c] == test:
					badaflag = 1
					break;

		# same check as above but for cells on same column as pointtocheck
		if badaflag == 0:
			for r in range(GRIDSIZE):
				if puzzle[r][pointtocheck[1]] == test:
					badaflag = 1
					break;

		# found a valid value, set test=10 to break out of loop and return this valid choice
		if badaflag == 0:
			assignment = test
			test = 10
		# otherwise update cell's invalid vector to indicate this choice of value has been found to be invalid
		else:
			invalid[test-1] = 1
	# return our choice of assignment (0 if found none) and the update invalid vector for the cell pointtocheck
	return assignment, invalid

### findnextvalid - Improved Version for forward checking ###
# This is findnextvalid for the Sudoku solver using foward checking
# As with findnextzero function only annotating things that have changed from previous version
# No longer pass invalid vector as this info is instead stored in the global invalidmatrix
def findnextvalidfwdcheck(puzzle, pointtocheck):
	global callcounter
	global backtrackcounter
	global invalidmatrix
	assignment = 0
	test = 0
	badaflag = 0
	while test <= 8:
		test += 1
		badaflag = 0
		# checking != 0 instead of ==1 because now using 1 and 2 to seperate invalid indicators made from forward checking (2) versus other sources (1)
		if invalidmatrix[pointtocheck[0]][pointtocheck[1]][test-1] != 0:
			badaflag = 1

		if badaflag == 0:
			for c in range(GRIDSIZE):
				if puzzle[pointtocheck[0]][c] == test:
					badaflag = 1
					break;
		if badaflag == 0:
			for r in range(GRIDSIZE):
				if puzzle[r][pointtocheck[1]] == test:
					badaflag = 1
					break;

		if badaflag == 0:
			assignment = test
			test = 10
		else:
			invalidmatrix[pointtocheck[0]][pointtocheck[1]][test-1] = 1
	return assignment

### findnextvalid - Improved Version for Sudoku solver using forward checking and least constraining value heuristic ###
def findnextvalidheuristics(puzzle, pointtocheck):
	global callcounter
	global backtrackcounter
	global invalidmatrix
	assignment = 0
	test = 0
	badaflag = 0
	# minconflicts introduced for least constraining value heuristic, 1000 is default so first valid choice will do better than it always
	minconflicts = 1000
	while test <= 8:
		test += 1
		badaflag = 0
		if invalidmatrix[pointtocheck[0]][pointtocheck[1]][test-1] != 0:
			badaflag = 1

		if badaflag == 0:
			for c in range(GRIDSIZE):
				if puzzle[pointtocheck[0]][c] == test:
					badaflag = 1
					break;
		if badaflag == 0:
			for r in range(GRIDSIZE):
				if puzzle[r][pointtocheck[1]] == test:
					badaflag = 1
					break;
		if badaflag == 0:
			testconflicts = 0
			# Least constraining value heuristic: only choose assignment if the number of unassigned cells (with value 0) on the same row or column as pointtocheck is less than our current best
			for i in range(GRIDSIZE):
				if invalidmatrix[i][pointtocheck[1]][test-1] == 0:
					testconflicts += 1
				if invalidmatrix[pointtocheck[0]][i][test-1] == 0:
					testconflicts += 1
			if testconflicts < minconflicts: 
				assignment = test
		else:
			invalidmatrix[pointtocheck[0]][pointtocheck[1]][test-1] = 1
	return assignment

### Sudoku Solver Main Function - Basic Backtracking Search Version ###
# puzzle is the grid and nextzero is the first row to check for unassigned cells (used to save some unneccesary checking in findnextzero)
def sudokusolve(puzzle, nextzero):
	global callcounter
	global backtrackcounter
	global maxiter
	global solution
	# setup invalid vector for the unassigned cell dealt with in this instance of sudokusolve
	invalid = [0] * 9
	# find next unassigned cell (with value 0)
	nextpt = findnextzero(puzzle, nextzero)
	# if findnextzero returns [-1,-1] then there are no more unassigned values, we have found the solution so set our current puzzle so set our current puzzle as the solution
	# and return 0 to indicate to parent function solution has been found (or driver if this is on top of the stack)
	if nextpt[0] == -1 and nextpt[1] == -1:
		solution = puzzle
		return 0
	while invalid != ([-1] * GRIDSIZE):
		# decrement iterations allowed
		maxiter -= 1
		# if reached max iterations return -2 up stack
		if maxiter == 0:
			return -2
		# find next valid digit/value for unassigned cell returned by findnextzero
		assignment, invalid = findnextvalid(puzzle, nextpt, invalid)
		# update puzzle to reflect valid assignment
		puzzle[nextpt[0]][nextpt[1]] = assignment
		# if assignment == 0 then there were no value assignments, send -1 up stack
		if assignment == 0:
			return -1
		# otherwise valid assignment made
		else:
			# increment callcounter
			callcounter += 1
			# call sudokusolve on updated puzzle (with additional info on what row to look for unassigned cells next in nextpt[0])
			callresult = sudokusolve(puzzle,nextpt[0])
			# pass info that solution found up stack
			if callresult == 0:
				return 0
			# pass info that max iterations reached up stack
			elif callresult == -2:
				return -2
			# backtracking - next instance of sudokusolve returned -1, meaning it couldn't find any valid assignments so we mark value we tried before as invalid and try again
			elif callresult == -1:
				invalid[assignment-1] = 1
				backtrackcounter += 1

### Sudoku Solver - Improved Version with Forward Checking ####
# see sudokusolve comments for shared info
def sudokusolvefwdcheck(puzzle, nextzero):
	global callcounter
	global backtrackcounter
	global invalidmatrix
	global maxiter
	global firsttimeflag
	global solution
	# this section initializes our invalidmatrix for forward checking with info on conflicts already present with initial values given in puzzle - only call this once
	if firsttimeflag == 1:
		for r in range(GRIDSIZE):
			for c in range (GRIDSIZE):
				pretest = puzzle[r][c]
				if pretest != 0:
					for i in range(GRIDSIZE):
						invalidmatrix[r][i][pretest-1] = 1
					for j in range(GRIDSIZE):
						invalidmatrix[j][c][pretest-1] = 1
		firsttimeflag = 0
	# find next unassigned cell, nextptr (next point row) and nextptc (next point column) are row/column of cell found by findnextzero
	nextpt = findnextzero(puzzle, nextzero)
	nextptr = nextpt[0]
	nextptc = nextpt[1]
	if nextptr == -1 and nextptc == -1:
		solution = puzzle
		return 0
	# this flag will flip and break the while loop if the current cell's invalid vector is all -1 (same as in basic version)
	flaginvalidallnegative = 0
	while flaginvalidallnegative == 0:
		maxiter -= 1
		if maxiter == 0:
			return -2
		assignment = findnextvalidfwdcheck(puzzle, nextpt)
		puzzle[nextpt[0]][nextpt[1]] = assignment
		forwardcheckingflag = 0

		# go through the invalidmatrix updating conflicts resulting from last assignment over all cells sharing row/column as current cell
		# (using value 2 instead of 1 so we can safely undo this as needed in algorithm without losing other info)
		if assignment != 0:
			for c in range(GRIDSIZE):
				if c != nextptc and puzzle[nextptr][c] == 0:
					if invalidmatrix[nextptr][c][assignment-1] == 0:
						invalidmatrix[nextptr][c][assignment-1] = 2
					# after updating invalidmatrix check cells on same row (same column in next block) as updated cell again
					# if there are no longer any valid assignments then flip fowardcheckingflag
					# to indicate later updated cell's assignment leads to another cell having no valid assignments of digits and so it is in turn invalid
					anyavailiable = 0
					for l in range(GRIDSIZE):
						if invalidmatrix[nextptr][c][l] == 0:
							anyavailiable = 1
					if anyavailiable == 0:
						forwardcheckingflag = 1
			# only check same thing for columns if rows did not lead to cell with no valid assignments
			if forwardcheckingflag == 0:
				for r in range(GRIDSIZE):
					if r != nextptr and puzzle[r][nextptc] == 0:
						if invalidmatrix[r][nextptc][assignment-1] == 0:
							invalidmatrix[r][nextptc][assignment-1] = 2
						anyavailiable = 0
						for l in range(GRIDSIZE):
							if invalidmatrix[r][nextptc][l] == 0:
								anyavailiable = 1
						if anyavailiable == 0:
							forwardcheckingflag = 1
		if assignment == 0:
			for i in range(GRIDSIZE):
				invalidmatrix[nextptr][nextptc][i] = 0

			return -1
		else:
			callresult = -10
			# 
			if(forwardcheckingflag==0):
				callcounter += 1
				callresult = sudokusolvefwdcheck(puzzle,nextpt[0])
				if callresult == 0:
					return 0
				elif callresult == -2:
					return -2
			# if either the fowardchecking failed (i.e. assignment not valid because other cell no longer has any valid assignments)
			# or child sudokusolve returned -1 meaning it found no valid assignments
			# then undo effects of forward checking (reset 2 to 0 in invalidmatrix for cells sharing row or column) if applicable
			# then mark the choice we made as invalid and try again with a different digit
			if(callresult == -1 or forwardcheckingflag == 1):
				for c in range(GRIDSIZE):
					if invalidmatrix[nextptr][c][assignment-1] == 2:
						invalidmatrix[nextptr][c][assignment-1] = 0
				for r in range(GRIDSIZE):
					if invalidmatrix[r][nextptc][assignment-1] == 2:
						invalidmatrix[r][nextptc][assignment-1] = 0
				invalidmatrix[nextptr][nextptc][assignment-1] = 1
				backtrackcounter += 1
		flaginvalidallnegative = 1
		for i in range(GRIDSIZE):
			if invalidmatrix[nextptr][nextptc][i] != -1:
				flaginvalidallnegative = 0

### Sudoku Solver - Final Version with Fowarding Checking and using least constrained value, most constrained variable and most constraining variable heuristics ###
# Same as previous version but calls the improved findnextzeroheuristics and findnextvalidheuristics
def sudokusolveheuristics(puzzle):
	global callcounter
	global backtrackcounter
	global invalidmatrix
	global maxiter
	global firsttimeflag
	global solution
	if firsttimeflag == 1:
		for r in range(GRIDSIZE):
			for c in range (GRIDSIZE):
				pretest = puzzle[r][c]
				if pretest != 0:
					for i in range(GRIDSIZE):
						invalidmatrix[r][i][pretest-1] = 1
					for j in range(GRIDSIZE):
						invalidmatrix[j][c][pretest-1] = 1
		firsttimeflag = 0
	nextpt = findnextzeroheuristics(puzzle)
	nextptr = nextpt[0]
	nextptc = nextpt[1]
	if nextptr == -1 and nextptc == -1:
		solution = puzzle
		return 0
	flaginvalidallnegative = 0
	while flaginvalidallnegative == 0:
		maxiter -= 1
		if maxiter == 0:
			return -2
		assignment = findnextvalidheuristics(puzzle, nextpt)
		puzzle[nextpt[0]][nextpt[1]] = assignment
		forwardcheckingflag = 0

		if assignment != 0:
			for c in range(GRIDSIZE):
				if c != nextptc and puzzle[nextptr][c] == 0:
					if invalidmatrix[nextptr][c][assignment-1] == 0:
						invalidmatrix[nextptr][c][assignment-1] = 2
					anyavailiable = 0
					for l in range(GRIDSIZE):
						if invalidmatrix[nextptr][c][l] == 0:
							anyavailiable = 1
					if anyavailiable == 0:
						forwardcheckingflag = 1
			
			if forwardcheckingflag == 0:
				for r in range(GRIDSIZE):
					if r != nextptr and puzzle[r][nextptc] == 0:
						if invalidmatrix[r][nextptc][assignment-1] == 0:
							invalidmatrix[r][nextptc][assignment-1] = 2
						anyavailiable = 0
						for l in range(GRIDSIZE):
							if invalidmatrix[r][nextptc][l] == 0:
								anyavailiable = 1
						if anyavailiable == 0:
							forwardcheckingflag = 1
		if assignment == 0:
			for i in range(GRIDSIZE):
				invalidmatrix[nextptr][nextptc][i] = 0

			return -1
		else:
			callresult = -10
			if(forwardcheckingflag==0):
				callcounter += 1
				callresult = sudokusolveheuristics(puzzle)
				if callresult == 0:
					return 0
				elif callresult == -2:
					return -2
			if(callresult == -1 or forwardcheckingflag == 1):
				for c in range(GRIDSIZE):
					if invalidmatrix[nextptr][c][assignment-1] == 2:
						invalidmatrix[nextptr][c][assignment-1] = 0
				for r in range(GRIDSIZE):
					if invalidmatrix[r][nextptc][assignment-1] == 2:
						invalidmatrix[r][nextptc][assignment-1] = 0
				invalidmatrix[nextptr][nextptc][assignment-1] = 1
				backtrackcounter += 1
		flaginvalidallnegative = 1
		for i in range(GRIDSIZE):
			if invalidmatrix[nextptr][nextptc][i] != -1:
				flaginvalidallnegative = 0

# resets global variables after call of sudoku solve
def resetglobals():
	global callcounter
	global backtrackcounter
	global invalidmatrix
	global maxiter
	global firsttimeflag
	global solution
	callcounter = 0
	backtrackcounter = 0
	invalidmatrix = np.zeros((GRIDSIZE, GRIDSIZE, GRIDSIZE))
	maxiter = 10000
	firsttimeflag = 1
	solution = []

# lists to store all results of call counter from multiple runs of sudokusolve over all 3 versions for different test instances
abasiccounter = []
afwdcheckcounter = []
aheuristicscounter = []

# lists to store results of call counter average over 10 examples per number of initial variables
avgbasiccounter = []
avgfwdcounter = []
avgheuristicscounter = []

readgrid = []
# how many initial numbers in sudoku problem
givennumbers = 1
# which of 10 problems per number of initial values given to choose
instance = 1

# iterate through all test data
for givennumbers in range(1,72):
	# used for average number of assignments per number of initial variables
	basicresultvec = []
	fwdcheckresultvec = []
	heuristicsresultvec = []
	avgresult = 0
	for instance in range (1,11):
		readgrid = []
		# read from text file
		with open('problems/' + str(givennumbers) +'/' + str(instance) +'.sd') as f:
			for i in range(GRIDSIZE):
				readgrid.append(next(f))
				row = []
				row = (readgrid[i]).split()
				for j in range(GRIDSIZE):
					row[j] = int(row[j])
				readgrid[i] = row

			# determine the very first 0 to pass initial row to check for 0 in first 2 versions of sudoku solve
			firstnextzero = findnextzero(readgrid, 0)

			# create copy of grid from file
			grid = copy.deepcopy(readgrid)
			# run sudokusolve
			result = sudokusolve(grid, firstnextzero[0])
			# print info on final solution and callcounter
			print("---Basic---")
			print("Number of Variable Assignments: " + str(callcounter))
			abasiccounter.append(callcounter)
			basicresultvec.append(callcounter)
			print("Solution:")
			print("---------------------------")
			# failure because reached maximum iterations
			if result == -2:
				print("Reached maximum iterations")
			# pathological case for debugging
			elif solution == []:
				print("Didn't reach max iterations but also failed...")
				print(solution)
				print(readgrid)
			# otherwise found solution - print it
			else:
				for printi in range(GRIDSIZE):
					print(solution[printi])
			print("---------------------------")
			print("\n")
			resetglobals()

			# same as above for version with forward checking
			grid = copy.deepcopy(readgrid)
			result = sudokusolvefwdcheck(grid, firstnextzero[0])
			print("---Forward Checking---")
			print("Number of Variable Assignments: " + str(callcounter))
			afwdcheckcounter.append(callcounter)
			fwdcheckresultvec.append(callcounter)
			print("Solution:")
			print("---------------------------")
			if result == -2:
				print("Reached maximum iterations")
			elif solution == []:
				print("Didn't reach max iterations but also failed...")
				print(solution)
				print(readgrid)
			else:
				for printi in range(GRIDSIZE):
					print(solution[printi])
			print("---------------------------")
			print("\n")
			resetglobals()

			# version with forward checking and heuristics
			grid = copy.deepcopy(readgrid)
			result = sudokusolveheuristics(grid)
			print("---Heuristics + Forward Checking---")
			print("Number of Variable Assignments: " + str(callcounter))
			aheuristicscounter.append(callcounter)
			heuristicsresultvec.append(callcounter)
			print("Solution:")
			print("---------------------------")
			if result == -2:
				print("Reached maximum iterations")
			elif solution == []:
				print("Didn't reach max iterations but also failed...")
				print(solution)
				print(readgrid)
			else:
				for printi in range(GRIDSIZE):
					print(solution[printi])
			print("---------------------------")
			print("\n")
			resetglobals()

	# Calculate and record averages for each initial element count
	avgresult = 0
	for element in basicresultvec:
		avgresult += element
	avgresult = avgresult/(len(basicresultvec))
	avgbasiccounter.append(avgresult)
	avgresult = 0
	for element in fwdcheckresultvec:
		avgresult += element
	avgresult = avgresult/(len(basicresultvec))
	avgfwdcounter.append(avgresult)
	avgresult = 0
	for element in heuristicsresultvec:
		avgresult += element
	avgresult = avgresult/(len(basicresultvec))
	avgheuristicscounter.append(avgresult)
	avgresult = 0

# Convert lists of call counters from each test instance into numpy arrays
plotbasiccounter = np.asarray(abasiccounter)
plotfwdcheckcounter = np.asarray(afwdcheckcounter)
plotheuristicscounter = np.asarray(aheuristicscounter)
plotavgbasiccounter = np.asarray(avgbasiccounter)
plotavgfwdcheckcounter = np.asarray(avgfwdcounter)
plotavgheuristicscounter = np.asarray(avgheuristicscounter)

# Set up index array for plotting all results
plotindex = np.arange(len(plotbasiccounter))

# Set up index array for plotting average per initial elements results
avgplotindex = np.arange(len(plotavgbasiccounter))

# Plot averaged results
lines = plt.plot(plotindex, abasiccounter, plotindex, afwdcheckcounter, plotindex, aheuristicscounter)
plt.xlim([0,70])
plt.ylim([-5,500])
plt.setp(lines[0], linewidth=1)
plt.setp(lines[1], linewidth=1)
plt.setp(lines[2], linewidth=1)
plt.legend(('Basic', 'Forward Checking', 'Heuristics + Forward Checking'), loc='upper left')
plt.title('Comparison of Sudoku Solvers')
plt.xlabel('Number of Initial Values')
plt.ylabel('Number of Variable Assignments')
plt.show()

# Plot raw results
lines = plt.plot(plotindex, abasiccounter, plotindex, afwdcheckcounter, plotindex, aheuristicscounter)
plt.xlim([0,750])
plt.ylim([-5,500])
plt.setp(lines[0], linewidth=1)
plt.setp(lines[1], linewidth=1)
plt.setp(lines[2], linewidth=1)
plt.legend(('Basic', 'Forward Checking', 'Heuristics + Forward Checking'), loc='upper left')
plt.title('Comparison of Sudoku Solvers')
plt.xlabel('Number of Initial Values (= floor(x/10))')
plt.ylabel('Number of Variable Assignments')
plt.show()

