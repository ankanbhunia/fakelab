import os
from IPython.display import HTML, clear_output
from IPython.display import Javascript
from IPython.display import Image
from google.colab.output import eval_js
import argparse

parser = argparse.ArgumentParser(description='FakeLab Options')

parser.add_argument('ngrok_auth_token', type=str, nargs='?',
                    help='Enter ngrok Authtoken from https://dashboard.ngrok.com/auth/your-authtoken ')

parser.add_argument('--path', type=str,
                    help='Specify drive path')
                    
parser.add_argument('--no_output', action='store_true',
                    help='the shell output will be disabled')

args = parser.parse_args()

xxx = args.ngrok_auth_token
drive_path = args.path
no_output_ = args.no_output


fakelab_ = """


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
    
import sys
clear_output()

ipy = get_ipython()
ipy.magic("tensorflow_version 1.x")

clear_output()
#import tensorflow as tf

#if not tf.test.is_gpu_available():
#  print ('Please use GPU to run the program.')
#  sys.exit(0)
  

GPU = get_ipython().getoutput("nvidia-smi --query-gpu=name --format=csv,noheader")

try:
  gpu = GPU[0]
except:
  gpu = 'CPU'

print (fakelab_)

if not os.path.isfile('/tmp/done'):
  if not os.path.isdir('/content/drive/'):
    from google.colab import drive; drive.mount('/content/drive', force_remount=True) 
    
    clear_output()
    
    
    print (fakelab_)

  print ('['+gpu+']'+' Please wait for few minutes... ')
  get_ipython().system_raw('git clone https://github.com/ankanbhunia/fakelab.git foo; mv foo/* foo/.git* .; rmdir foo; gdown --id 1-lLw4WSCfP7wYsk3-6Xv4m0I3aUBjMzJ; tar -xvf fake-lab-lib-v1.0.tar.gz; rm fake-lab-lib-v1.0.tar.gz; touch /tmp/done')
  get_ipython().system_raw('sudo apt-get install -y xattr')

clear_output()

print (fakelab_)

print ("[GPU Device]="+gpu)


print ("""

""")

if xxx:

  try:

    get_ipython().system_raw("pip3 install pyngrok")
    get_ipython().system_raw("ngrok authtoken " + xxx)
    from pyngrok import ngrok
    ngrok.kill()
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

get_ipython().system_raw("fuser -k 8000/tcp")

if drive_path:

    if no_output_:
      get_ipython().system_raw("Library/bin/python app.py "+drive_path)
    else:
      G = get_ipython().getoutput("Library/bin/python app.py "+drive_path)
      
else:

    if no_output_:
      get_ipython().system_raw("Library/bin/python app.py")
    else:
      G = get_ipython().getoutput("Library/bin/python app.py")
