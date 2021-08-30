#import subprocess32
import pandas as pd

command = 'cd /Users/dimi/Desktop/SPIDER3-S; ./impute_script_np.sh'
ssprocess = subprocess32.Popen(command, shell=True, stdout=subprocess32.PIPE, stderr=subprocess32.STDOUT)
ssout, sserr = ssprocess.communicate()

kp32ss = pd.read_csv('/Users/dimi/Desktop/SPIDER3-S/example_data/outputs/KP32_2.i1', delim_whitespace=True, header=0)
sstructs = ''.join([i for i in kp32ss['SS']])