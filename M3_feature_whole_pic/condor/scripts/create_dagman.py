#!/usr/bin/env python3

# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""generates a list of dagman jobs, by combining a set of variables with its ranges."""

import numpy as np

TEMPLATE_JOB = 'JOB {} retipy.condor\n'
TEMPLATE_VARIABLES = '''VARS {0} w="{1}"
VARS {0} ppw="{2}"
VARS {0} ss="{3}"
VARS {0} r2t="{4:1.2f}"'''

CURRENT_JOB = 1
JOBS = ""
VARS = ""

for w in range(8, 128, 8):  # window size
    for ppw in range(4, 10, 2):  # pixels per window
        for ss in range(4, 6, 1):  # sampling size
            for r2t in np.arange(0.40, 0.99, 0.01):  #r2 threshold
                JOBS += TEMPLATE_JOB.format('RETIPY'+str(CURRENT_JOB))
                VARS += TEMPLATE_VARIABLES.format('RETIPY'+str(CURRENT_JOB), w, ppw, ss, r2t)
                VARS += '\n'
                CURRENT_JOB += 1

FILE = open('retipy.dag', 'w')
FILE.write(JOBS)
FILE.write('\n')
FILE.write(VARS)
