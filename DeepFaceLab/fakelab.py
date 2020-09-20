import os
from IPython.display import HTML, clear_output
from IPython.display import Javascript
from IPython.display import Image
from google.colab.output import eval_js
import argparse

parser = argparse.ArgumentParser(description='FakeLab Options')

parser.add_argument('ngrok_auth_token', type=str, nargs='?',
                    help='Enter ngrok Authtoken from https://dashboard.ngrok.com/auth/your-authtoken ')

parser.add_argument('--no_output', action='store_true',
                    help='the shell output will be disabled')

args = parser.parse_args()

xxx = args.ngrok_auth_token
no_output_ = args.no_output

import sys
clear_output()

ipy = get_ipython()
ipy.magic("tensorflow_version 1.x")

import tensorflow as tf

if not tf.test.is_gpu_available():
  print ('Please use GPU to run the program.')
  sys.exit(0)
  

GPU = get_ipython().getoutput("nvidia-smi --query-gpu=name --format=csv,noheader")

try:
  gpu = GPU[0]
except:
  gpu = 'CPU'

if not os.path.isfile('/tmp/done'):
  if not os.path.isdir('/content/drive/'):
    from google.colab import drive; drive.mount('/content/drive', force_remount=True) 
    
  clear_output()

  print ('['+gpu+']'+' Please wait for few minutes... ')
  get_ipython().system_raw('git clone https://github.com/ankanbhunia/deeplabs ; cp -r deeplabs/* /content; rm -r deeplabs; python install_.py; touch /tmp/done')



clear_output()

print ("""


  .-.            ___               ___         ___      
 /    \         (   )             (   )       (   )     
 | .`. ;  .---.  | |   ___   .--.  | |   .---. | |.-.   
 | |(___)/ .-, \ | |  (   ) /    \ | |  / .-, \| /   \  
 | |_   (__) ; | | |  ' /  |  .-. ;| | (__) ; ||  .-. | 
(   __)   .'`  | | |,' /   |  | | || |   .'`  || |  | | 
 | |     / .'| | | .  '.   |  |/  || |  / .'| || |  | | 
 | |    | /  | | | | `. \  |  ' _.'| | | /  | || |  | | 
 | |    ; |  ; | | |   \ \ |  .'.-.| | ; |  ; || '  | | 
 | |    ' `-'  | | |    \ .'  `-' /| | ' `-'  |' `-' ;  
(___)   `.__.'_.(___ ) (___)`.__.'(___)`.__.'_. `.__.   
                                                                                                               
"""
)

print ("[GPU Device]="+gpu)


print ("""

""")

if xxx:

  try:

    get_ipython().system_raw("pip install pyngrok")
    get_ipython().system_raw("ngrok authtoken " + xxx)
    from pyngrok import ngrok
    print("Project URL: "+ngrok.connect(port = '8000'))

  except:

    print ('The ngrok token is invalid')
    print("Project URL: "+eval_js("google.colab.kernel.proxyPort(%d)"% (8000)))

else:

  print("Project URL: "+eval_js("google.colab.kernel.proxyPort(%d)"% (8000)))


print("""
                                                                                                                                                                                                 
                                                                                                                                           
  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______ 
 |______||______||______||______||______||______||______||______||______||______||______||______||______||______||______||______||______||______||______||______||______|
                                                                                                                                                                         
""")

if no_output_:
  get_ipython().system_raw("python3 app.py")
else:
  G = get_ipython().getoutput("python3 app.py")
