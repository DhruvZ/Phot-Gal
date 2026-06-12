import os
import sys
#path = os.path.join(os.path.dirname(__file__), 'all')
#print(path)

sys.path.append(os.path.dirname(__file__))

from . import phot_gal_object_release
from . import run_fit


