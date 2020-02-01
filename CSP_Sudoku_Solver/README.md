=================
CSP Sudoku Solver
=================
This project implements several constraint satisfaction problem algorithms to solve Sudoku puzzles. It also analyzes the performance improvement (measured as the number of variable assignments) of more advanced versions of the basic backtracking search by adding forward checking and various heuristics including most constraining variable, most constrained variable and least constrained value.

=================
Prerequisites
=================
The algorithms in sudoku.py require numpy.
The plotting code requires matplotlib and copy.

=================
Details
=================

Three versions of a constraint satisfaction problem algorithm for Sudoku are implemented in sudoku.py. One using just backtracking search, a second version using forward checking as well to reduce the number of variable assignments and lastly a complete version using both forward checking and heuristics.

puzzle.sd is a standard input example that the algorithm can read and solve.
puzzle_needs_forward_checking.sd is an edge case example that requires that the basic solver cannot feasibly complete.
puzzle_needs_heuristics.sd is an edge case example that only the complete algorithm with forward checking and heuristics can solve in a reasonable amount of time.
Rename either tests to puzzle.sd to test them by running sudoku.py.

The Problems folder contains 710 examples problems with subdirectories for problems by number of initial values given (1-71).
Rename any example puzzle.sd and place it in the same directory as sudoku.py then run sudoku.py to test on that example.

sudoku_plot.py runs through every problem in the Problems directory with each of the three versions of the algorithm plotting the raw and normalized results using matplotlib.
The x-axis is the number of initial values assigned in the example and the y-axis is the number of variable assignments needed to solve the puzzle.

linear_regression_plot.py performs normalization and linear regression on the results of testing the algorithms on all 710 examples to produce a plot that clearly illustrates the performance improvement of forward checking and the various heuristics.

results.txt lists the raw results from running all three versions on the examples from Problems.

average_per_initial_value_count.png is one of the resulting plots from running sudoku_plot.py. Showing the average number of variable assignments needed to solve a puzzle given 1-71 initial values.

linear_regression.png is the resulting plot from running linear_regression_plot.py comparing the three versions of the Sudoku solver across all the examples in Problems.

=================
Acknowledgements
=================
The test problems are courtesy of the University of Waterloo



