#! /usr/bin/env python
"""
Sort CSV file by multiple columns, writing output to sorted CSV file.
Recommended for files saved in Windows CSV format.
Useful for situations where data file is too large for Excel.

: param source_file.csv : source csv file. Must end in .csv 
: param sort column 1 : first sort in Excel-like column number (i.e., 1 ... N)
                        Use negative number to indicate descending sort,
                        Positive number to indicate ascending sort_step
: param sort column N : next sort in Excel-like column number (i.e., 1 ... N)

Result is a sorted destination csv of name format input_file_sorted.csv.

EXAMPLES: 
	foo.csv has Column 1 = last_name, Column 2 = first_name
				Column 3 = dob, Column 4 = height
	Sort foo.csv by last_name then first_name then DOB:
		sortCSV.py foo.csv 1 2 3
	Sort the same but sort DOB descending (vs ascending):
		sortCSV.py foo.csv 1 2 -3
	Sort foo.csv by last_name then first_name then height, tallest to short:
		sortCSV.py foo.csv 1 2 -4
	Output written to foo_sorted.csv

Move to /usr/local/bin and chmod +x to use as command.
Can easily convert to function for real-time application.
"""

import sys
import csv
from sys import argv
from operator import itemgetter

num_arguments = len(argv)

# Check usage and provide help
if num_arguments == 2 and argv[1] in ('-h', '-help'):
	print("Usage: %s input_file.csv 1st_sort_col ... nth_sort_col" % argv[0])
	print("Example: %s foo.csv 1 2 -9" % argv[0])
	print("\tSorts foo.csv on 1st and 2nd columns (ascending) then 9th descending.")
	sys.exit()
elif num_arguments < 3: # Guidance on arguments to pass
	usage = "Usage: %s input_file.csv 1st_sort_col ... nth_sort_col" % argv[0]
	error = "You passed only %d arguments" % num_arguments
	sys.exit("%s -- %s" % (usage, error))
if '.csv' not in argv[1]: # Ensure using a CSV file
	usage = "Usage: %s input_file.csv 1st_sort_col ... nth_sort_col" % argv[0]
	error = "You passed %r for input_file.csv" % argv[1]
	sys.exit("%s -- %s" % (usage, error))

# Create the output file as input with _sorted before .csv extension
input_file = argv[1]
output_file = input_file.replace('.csv', '_sorted.csv')

# Ensure you can open the source and target files
try:
	source = open(input_file, 'r')
except:
	e = sys.exc_info()[0]
	sys.exit("Error - Could not open input file %r: %s" % (input_file, e))
try: 
	target = open(output_file, 'w')
except:
	e = sys.exc_info()[0]
	sys.exit("Error - Could not open output file %r: %s" % (output_file, e))
print("\nSorting data from %r into %r, inside out" % (input_file, output_file))

# Now create a list of sorting tuples where the first item is the index of 
# the data object you wish to sort and the second item is the type of sort,
# Ascending (Reverse is False) or Descending (Reverse is True)
sorts = []
for i in range (2, num_arguments): # Skip script name and input filename
	# Ensure you are passed Excel-like column numbers
	try: 
		sort_arg = int(argv[i])
	except:
		e = sys.exc_info()[0]
		sys.exit("Error - Sort column %r not an integer: %s." % (argv[i], e))
	if sort_arg == 0:
		sys.exit("Error - Use Excel-like column numbers from 1 to N")
	# Create a tuple for each as described above
	if sort_arg > 0:
		sorts.append((sort_arg - 1, False)) # Convert column num to index num
	else:
		sorts.append(((-1 * sort_arg) - 1, True))

# Read in the data creating a label list and list of one tuple per row
reader = csv.reader(source)
row_count = 0 
data=[] 
for row in reader:
	row_count += 1
	# Place the first row into the header
	if row_count == 1:
		header = row
		continue
	# Append all non-header rows into a list of data as a tuple of cells
	data.append(tuple(row))

# Sort is stable as of Python 2.2. As such, we can break down the 
# complex sort into a series of simpler sorts. We just need to remember 
# to REVERSE the order of the sorts.
for sort_step in reversed(sorts):
	print('Sorting Column %d ("%s") Descending=%s' % \
	(sort_step[0] + 1, header[sort_step[0]], sort_step[1])) # +1 for Excel col num
	data = sorted(data, key=itemgetter(sort_step[0]), reverse=sort_step[1])
print('Done sorting %d data rows (excluding header row) from %r' % \
((row_count - 1), input_file))


# Now write all of this out to the new file
writer = csv.writer(target)
writer.writerow(header) # Write the header in CSV format
for sorted_row in data: # Wrtie the sorted data, converting to CSV format
	writer.writerow(sorted_row)
print('Done writing %d rows (sorted data plus header) to %r\n' % \
(row_count, output_file)) 

# Be good and close the files
source.closed
target.closed