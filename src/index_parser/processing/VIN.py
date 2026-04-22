import os
import time
import shutil
from pathlib import Path
import pandas as pd
from statistics import stdev, mean



#accepts 
def verify_intendation(coord_list):
    i_min_2, i_min_1, i, i_plus_1, i_plus_2 = coord_list

    negatives = sum(1 for x in [i_min_2, i_min_1, i_plus_1, i_plus_2] if x - i < -0.1)
    
    if negatives >= 2:
        return True
    else:
        return False


