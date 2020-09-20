
import os
import sys 



os.system('pip3 install --upgrade moviepy')
os.system('pip3 install dash')
os.system('pip3 install dash_core_components')
os.system('pip3 install dash_html_components')
os.system('pip3 install dash_bootstrap_components')
os.system('pip3 install flask')
os.system('pip3 install youtube_dl')
os.system('pip3 install mhyt')
os.system('pip3 install dash_daq')
os.system('pip3 install dash-editor-components')
#os.system('pip3 install scipy==1.1.0')

Mode = "install" 

from pathlib import Path
if (Mode == "install"):
  os.system('git clone https://github.com/ankanbhunia/DeepFaceLab')
  os.system('git clone https://github.com/ankanbhunia/motion-cosegmentation motion-co-seg')
  os.system('git clone https://github.com/ankanbhunia/face-makeup.PyTorch motion-co-seg/face_parsing')
  os.system('git clone http://github.com/ankanbhunia/FaceClust/')

  sys.path.append('/content/motion-co-seg/')
  sys.path.append('/content/DeepFaceLab/')

  # fix linux warning
  # /usr/lib/python3.6/multiprocessing/semaphore_tracker.py:143: UserWarning: semaphore_tracker: There appear to be 1 leaked semaphores to clean up at shutdown
  fin = open("/usr/lib/python3.6/multiprocessing/semaphore_tracker.py", "rt")
  data = fin.read()
  data = data.replace('if cache:', 'if False:')
  fin.close()

  fin = open("/usr/lib/python3.6/multiprocessing/semaphore_tracker.py", "wt")
  fin.write(data)
  fin.close()
else:

  os.system('/content/DeepFaceLab | git pull')

os.system('pip install -r /content/DeepFaceLab/requirements-colab.txt')
os.system('pip install --upgrade scikit-image')
os.system('apt-get install cuda-10-0')

if not Path("/content/pretrain").exists():
  print("Downloading CelebA faceset ... ")
  os.system('wget -q --no-check-certificate -r https://github.com/chervonij/DFL-Colab/releases/download/pretrain-CelebA/pretrain_CelebA.zip -O /content/pretrain_CelebA.zip')
  os.system('mkdir /content/pretrain')
  os.system('unzip -q /content/pretrain_CelebA.zip -d /content/pretrain/')
  os.system('rm /content/pretrain_CelebA.zip')

#if not Path("/content/pretrain_Q96").exists():
#  print("Downloading Q96 pretrained model ...")
#  os.system('wget -q --no-check-certificate -r https://github.com/chervonij/DFL-Colab/releases/download/Q96_model_pretrained/Q96_model_pretrained.zip -O /content/pretrain_Q96.zip')
#  os.system('mkdir /content/pretrain_Q96')
#  os.system('unzip -q /content/pretrain_Q96.zip -d /content/pretrain_Q96/')
#  os.system('rm /content/pretrain_Q96.zip')

if not Path("/content/workspace").exists():
  os.system('mkdir /content/workspace; mkdir /content/workspace/data_src; mkdir /content/workspace/data_src/aligned; mkdir /content/workspace/data_dst; mkdir /content/workspace/data_dst/aligned; mkdir /content/workspace/model')  


#print("\nDone!")
#os.system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
#os.system('unzip ngrok-stable-linux-amd64.zip')


#os.system('./ngrok authtoken 5vhWvAzJGtsJbnVp4V5di_6KNVTN8BpHMqKYyAaFFXQ')
print("Done!")