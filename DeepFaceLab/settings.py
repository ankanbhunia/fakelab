############################################
             #BASIC PARAMETERS
############################################

fps = 0 #How many frames of every second of the video will be extracted. 0 - full fps
Resolution = 256 # Face Resolution: [64-640] More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16 and 32 for -d archi.
Output_image_Resolution = 512 #[256,2048] Output image size. The higher image size, the worse face-enhancer works. Use higher than 512 value only if the source image is sharp enough and the face does not need to be enhanced
Face_Type = 'head' #['f','wf','head'] Full face / whole face / head. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.
Batch_size = 8 # If you get memory error reduce the batch size
Target_Iterations = 500000 # Specify total Iteration  
max_faces_from_image = 0 #Max number of faces from image:If you extract a src faceset that has frames with a large number of faces, it is advisable to set max faces to 3 to speed up extraction. 0 - unlimited
output_img_ext = "png" # ["png","jpg"] png is lossless, but extraction is x10 slower for HDD, requires x10 more disk space than jpg.
jpeg_quality = 90 #[1,100] Jpeg quality. The higher jpeg quality the larger the output file size.
bitrate = 16 # Bitrate of output video file in MB/s
output_debug = False #Write debug images
models_opt_on_gpu = True #Place models and optimizer on GPU

############################################
            #Training PARAMETERS
############################################

Denoise_factor = 1 #[1-20]
ae_dims = 256 #AutoEncoder dimensions: All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU.
e_dims = 64 #Encoder dimensions: More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU.
d_dims = None #Decoder dimensions: More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU.
d_mask_dims = None #Decoder mask dimensions: Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality.
masked_training = True #This option is available only for 'whole_face' or 'head' type. Masked training clips training area to full_face mask or XSeg mask, thus network will train the faces properly.
eyes_prio = False #Eyes priority: Helps to fix eye problems during training like "alien eyes" and wrong eyes direction ( especially on HD architectures ) by forcing the neural network to train eyes with higher priority. before/after
uniform_yaw = False #Uniform yaw distribution of samples: Helps to fix blurry side faces due to small amount of them in the faceset.
random_warp = True #Enable random warp of samples: Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.
gan_power = 0.0 #Train the network in Generative Adversarial manner. Accelerates the speed of training. Forces the neural network to learn small details of the face. Enable it only when the face is trained enough and don't disable. Typical value is 1.0
true_face_power = 0.0 #Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01
face_style_power = 0.0 #Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse.
bg_style_power = 0.0 #Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. For whole_face you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0
ct_mode = 'none' #Color transfer for src faceset: ['none','rct','lct','mkl','idt','sot']. Change color distribution of src samples close to dst samples. Try all modes to find the best.
clipgrad = False #Gradient clipping reduces chance of model collapse, sacrificing speed of training.
pretrain = False #Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly.

############################################
           #Conversion PARAMETERS
############################################

merging_mode = 1 #(0) original (1) overlay (2) hist-match (3) seamless (4) seamless-hist-match (5) raw-rgb (6) raw-predict
mask_merging_mode = 2 #(1) dst (2) learned-prd (3) learned-dst (4) learned-prd*learned-dst (5) learned-prd+learned-dst (6) XSeg-prd (7) XSeg-dst (8) XSeg-prd*XSeg-dst (9) learned-prd*learned-dst*XSeg-prd*XSeg-dst
sharpen_mode = 1 #(0) None (1) box (2) gaussian
blursharpen_amount = 0 # [-100,100]
erode_mask_modifier = 0 # [-400, 400] Choose erode mask modifier
blur_mask_modifier = 0 # [0, 400] Choose blur mask modifier
motion_blur_power = 0 # [0, 100] Choose motion blur power
output_face_scale = 0 # [-50, 50] Choose output face scale modifier
color_transfer_mode = None # [rct/lct/mkl/mkl-m/idt/idt-m/sot-m/mix-m]
super_resolution_power = 100 # [0,100] Choose super resolution power
image_denoise_power = 0 # [0,500] Choose image degrade by denoise power
bicubic_degrade_power = 0 # [0,100] Choose image degrade by bicubic rescale power
color_degrade_power = 0 # [0,100] Degrade color power of final image
masked_hist_match = True # Masked hist match?
hist_match_threshold = 255 #[0, 255]
horizontal_shear = 0 # face imposing geometry options
vertical_shear = 0 # face imposing geometry options
horizontal_shift = 0 # face imposing geometry options
vertical_shift = 0  # face imposing geometry options