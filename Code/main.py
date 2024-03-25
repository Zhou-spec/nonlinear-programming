import numpy as np
import time
import os
from runcase import *


    
def mainfunc_hw2():
    # Create a directory call result to store the result
    import os
    if not os.path.exists('result'):
        os.makedirs('result')
        
    methodlist = ['gradient_descend', 'Newton', 'modified_Newton']
    for i in range(10):
        runcase_with_id(i)
            

def mainfunc_hw3():
    # Create a directory call result to store the result
    import os
    if not os.path.exists('result'):
        os.makedirs('result')
        
    methodlist = ['BFGS', 'L-BFGS', 'Newton-CG']
    for i in range(10):
        runcase_with_id(i, methodlist)    
            
if __name__ == '__main__':
    mainfunc_hw3()


