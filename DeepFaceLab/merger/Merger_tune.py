import settings

import sys
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt

from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor
import glob
is_windows = sys.platform[0:3] == 'win'
xseg_input_size = 256
import settings
sys.path.append('/content/DeepFaceLab/')


from DFLIMG import DFLIMG
from pathlib import Path


def geometric_transformation_of_mask(img_bgr, img_face, mask, H = 0, V = 0, PH = 0, PV = 0): 

  r,c,ch = img_bgr.shape

  (tx,ty) = (PH, -PV)

  M = np.float32([[1,0,tx],[0,1,ty]])


  mask = cv2.warpAffine(mask,M,(c,r))
  img_face = cv2.warpAffine(img_face,M,(c,r))


  if H>=0: (H_P, H_N) = (H,0) 
  else: (H_P, H_N) = (0,-H)
  if V>=0: (V_P, V_N) = (V,0) 
  else: (V_P, V_N) = (0,-V)

  pts1 = np.float32([[H_P,r//2], [c//2,V_P],[c-H_P,r//2],  [c//2,r-V_P]])
  pts2 = np.float32([[H_N,r//2],  [c//2,V_N], [c-H_N,r//2],  [c//2,r-V_N]])

  M = cv2.getPerspectiveTransform(pts1,pts2)


  mask_ = cv2.warpPerspective(mask,M,(c,r))
  img_bgr_ = cv2.warpPerspective(img_bgr,M,(c,r))
  img_face_ = cv2.warpPerspective(img_face,M,(c,r))

  return img_bgr*(1-mask_[..., np.newaxis]) + (img_face_*mask_[..., np.newaxis])
  


def MergeMaskedFace_test(path, cfg):
  
    data = np.load(path, allow_pickle=True)

    [img_bgr, predictor_input_shape, frame_info, img_face_landmarks, prd_face_bgr, prd_face_mask_a_0, prd_face_dst_mask_a_0, img_bgr_uint8] = data
    
    img_path = glob.glob('/content/workspace/data_dst/aligned/'+path.split('/')[-1].split('_')[1]+'*')[0]
    dflimg = DFLIMG.load(Path(img_path))
    
    face_type_ = dflimg.get_face_type()
    if face_type_ == 'head':
    
        cfg.face_type = FaceType.HEAD
        
    elif face_type_ == 'f':
    
        cfg.face_type = FaceType.FULL
    
    elif face_type_ == 'wf':
    
        cfg.face_type = FaceType.WHOLE_FACE
        
    out_img = None

    if cfg.show_mode == 1 or cfg.show_mode ==3:
    
        img_size = img_bgr.shape[1], img_bgr.shape[0]
        img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)

        input_size = predictor_input_shape[0]
        mask_subres_size = input_size*4
        output_size = input_size
        if cfg.super_resolution_power != 0:
            output_size *= 4

        face_mat        = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=cfg.face_type)
        face_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=cfg.face_type, scale= 1.0 + 0.01*cfg.output_face_scale)

        if mask_subres_size == output_size:
            face_mask_output_mat = face_output_mat
        else:
            face_mask_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, mask_subres_size, face_type=cfg.face_type, scale= 1.0 + 0.01*cfg.output_face_scale)

        dst_face_bgr      = cv2.warpAffine(img_bgr, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
        dst_face_bgr      = np.clip(dst_face_bgr, 0, 1)

        dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
        dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

        predictor_input_bgr      = cv2.resize (dst_face_bgr, (input_size,input_size) )



        if cfg.mask_mode == 1: #dst
            wrk_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (output_size,output_size), interpolation=cv2.INTER_CUBIC)
        elif cfg.mask_mode == 2: #learned-prd
            wrk_face_mask_a_0 = prd_face_mask_a_0
        elif cfg.mask_mode == 3: #learned-dst
            wrk_face_mask_a_0 = prd_face_dst_mask_a_0
        elif cfg.mask_mode == 4: #learned-prd*learned-dst
            wrk_face_mask_a_0 = prd_face_mask_a_0*prd_face_dst_mask_a_0
        elif cfg.mask_mode == 5: #learned-prd+learned-dst
            wrk_face_mask_a_0 = np.clip( prd_face_mask_a_0+prd_face_dst_mask_a_0, 0, 1)
        elif cfg.mask_mode >= 6 and cfg.mask_mode <= 9:  #XSeg modes



            if cfg.mask_mode == 6:   #'XSeg-prd'
                wrk_face_mask_a_0 = X_prd_face_mask_a_0
            elif cfg.mask_mode == 7: #'XSeg-dst'
                wrk_face_mask_a_0 = X_dst_face_mask_a_0
            elif cfg.mask_mode == 8: #'XSeg-prd*XSeg-dst'
                wrk_face_mask_a_0 = X_prd_face_mask_a_0 * X_dst_face_mask_a_0
            elif cfg.mask_mode == 9: #learned-prd*learned-dst*XSeg-prd*XSeg-dst
                wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0 * X_prd_face_mask_a_0 * X_dst_face_mask_a_0

        wrk_face_mask_a_0[ wrk_face_mask_a_0 < (1.0/255.0) ] = 0.0 # get rid of noise

        # resize to mask_subres_size
        if wrk_face_mask_a_0.shape[0] != mask_subres_size:
            wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (mask_subres_size, mask_subres_size), interpolation=cv2.INTER_CUBIC)

        # process mask in local predicted space
        if 'raw' not in cfg.mode:
            # add zero pad
            wrk_face_mask_a_0 = np.pad (wrk_face_mask_a_0, input_size)

            ero  = cfg.erode_mask_modifier
            blur = cfg.blur_mask_modifier

            if ero > 0:
                wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
            elif ero < 0:
                wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )

            # clip eroded/dilated mask in actual predict area
            # pad with half blur size in order to accuratelly fade to zero at the boundary
            clip_size = input_size + blur // 2

            wrk_face_mask_a_0[:clip_size,:] = 0
            wrk_face_mask_a_0[-clip_size:,:] = 0
            wrk_face_mask_a_0[:,:clip_size] = 0
            wrk_face_mask_a_0[:,-clip_size:] = 0

            if blur > 0:
                blur = blur + (1-blur % 2)
                wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur) , 0)

            wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size,input_size:-input_size]

            wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)

        img_face_mask_a = cv2.warpAffine( wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(img_bgr.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )[...,None]
        img_face_mask_a = np.clip (img_face_mask_a, 0.0, 1.0)
        img_face_mask_a [ img_face_mask_a < (1.0/255.0) ] = 0.0 # get rid of noise

        if wrk_face_mask_a_0.shape[0] != output_size:
            wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (output_size,output_size), interpolation=cv2.INTER_CUBIC)

        wrk_face_mask_a = wrk_face_mask_a_0[...,None]


        out_merging_mask_a = None
        if cfg.mode == 'original':
            return img_bgr, img_face_mask_a

        elif 'raw' in cfg.mode:
            if cfg.mode == 'raw-rgb':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
                out_merging_mask_a = img_face_mask_a
            elif cfg.mode == 'raw-predict':
                out_img = prd_face_bgr
                out_merging_mask_a = wrk_face_mask_a
            else:
                raise ValueError(f"undefined raw type {cfg.mode}")

            out_img = np.clip (out_img, 0.0, 1.0 )
        else:

            # Process if the mask meets minimum size
            maxregion = np.argwhere( img_face_mask_a >= 0.1 )
            if maxregion.size != 0:
                miny,minx = maxregion.min(axis=0)[:2]
                maxy,maxx = maxregion.max(axis=0)[:2]
                lenx = maxx - minx
                leny = maxy - miny
                if min(lenx,leny) >= 4:
                    wrk_face_mask_area_a = wrk_face_mask_a.copy()
                    wrk_face_mask_area_a[wrk_face_mask_area_a>0] = 1.0

                    if 'seamless' not in cfg.mode and cfg.color_transfer_mode != 0:
                        if cfg.color_transfer_mode == 1: #rct
                            prd_face_bgr = imagelib.reinhard_color_transfer ( np.clip( prd_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8),
                                                                              np.clip( dst_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8), )

                            prd_face_bgr = np.clip( prd_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)
                        elif cfg.color_transfer_mode == 2: #lct
                            prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 3: #mkl
                            prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 4: #mkl-m
                            prd_face_bgr = imagelib.color_transfer_mkl (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 5: #idt
                            prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 6: #idt-m
                            prd_face_bgr = imagelib.color_transfer_idt (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 7: #sot-m
                            prd_face_bgr = imagelib.color_transfer_sot (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
                            prd_face_bgr = np.clip (prd_face_bgr, 0.0, 1.0)
                        elif cfg.color_transfer_mode == 8: #mix-m
                            prd_face_bgr = imagelib.color_transfer_mix (prd_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)

                    if cfg.mode == 'hist-match':
                        hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                        if cfg.masked_hist_match:
                            hist_mask_a *= wrk_face_mask_area_a

                        white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                        hist_match_1 = prd_face_bgr*hist_mask_a + white
                        hist_match_1[ hist_match_1 > 1.0 ] = 1.0

                        hist_match_2 = dst_face_bgr*hist_mask_a + white
                        hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                        prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, cfg.hist_match_threshold ).astype(dtype=np.float32)

                    if 'seamless' in cfg.mode:
                        #mask used for cv2.seamlessClone
                        img_face_seamless_mask_a = None
                        for i in range(1,10):
                            a = img_face_mask_a > i / 10.0
                            if len(np.argwhere(a)) == 0:
                                continue
                            img_face_seamless_mask_a = img_face_mask_a.copy()
                            img_face_seamless_mask_a[a] = 1.0
                            img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] = 0.0
                            break

                    out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )
                    out_img = np.clip(out_img, 0.0, 1.0)

                    if 'seamless' in cfg.mode:
                        try:
                            #calc same bounding rect and center point as in cv2.seamlessClone to prevent jittering (not flickering)
                            l,t,w,h = cv2.boundingRect( (img_face_seamless_mask_a*255).astype(np.uint8) )
                            s_maskx, s_masky = int(l+w/2), int(t+h/2)
                            out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), img_bgr_uint8, (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx,s_masky) , cv2.NORMAL_CLONE )
                            out_img = out_img.astype(dtype=np.float32) / 255.0
                        except Exception as e:
                            #seamlessClone may fail in some cases
                            e_str = traceback.format_exc()

                            if 'MemoryError' in e_str:
                                raise Exception("Seamless fail: " + e_str) #reraise MemoryError in order to reprocess this data by other processes
                            else:
                                print ("Seamless fail: " + e_str)

                    cfg_mp = cfg.motion_blur_power / 100.0

                    out_img = geometric_transformation_of_mask(img_bgr, out_img, img_face_mask_a, H = cfg.horizontal_shear, V = cfg.vertical_shear, PH =  cfg.horizontal_shift, PV = cfg.vertical_shift)

                    if ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
                       cfg.mode == 'seamless-hist-match' or \
                       cfg_mp != 0 or \
                       cfg.blursharpen_amount != 0 or \
                       cfg.image_denoise_power != 0 or \
                       cfg.bicubic_degrade_power != 0:

                        out_face_bgr = cv2.warpAffine( out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )

                        if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
                            if cfg.color_transfer_mode == 1:
                                out_face_bgr = imagelib.reinhard_color_transfer ( np.clip(out_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8),
                                                                                  np.clip(dst_face_bgr*wrk_face_mask_area_a*255, 0, 255).astype(np.uint8) )
                                out_face_bgr = np.clip( out_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)
                            elif cfg.color_transfer_mode == 2: #lct
                                out_face_bgr = imagelib.linear_color_transfer (out_face_bgr, dst_face_bgr)
                            elif cfg.color_transfer_mode == 3: #mkl
                                out_face_bgr = imagelib.color_transfer_mkl (out_face_bgr, dst_face_bgr)
                            elif cfg.color_transfer_mode == 4: #mkl-m
                                out_face_bgr = imagelib.color_transfer_mkl (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                            elif cfg.color_transfer_mode == 5: #idt
                                out_face_bgr = imagelib.color_transfer_idt (out_face_bgr, dst_face_bgr)
                            elif cfg.color_transfer_mode == 6: #idt-m
                                out_face_bgr = imagelib.color_transfer_idt (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)
                            elif cfg.color_transfer_mode == 7: #sot-m
                                out_face_bgr = imagelib.color_transfer_sot (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a, steps=10, batch_size=30)
                                out_face_bgr = np.clip (out_face_bgr, 0.0, 1.0)
                            elif cfg.color_transfer_mode == 8: #mix-m
                                out_face_bgr = imagelib.color_transfer_mix (out_face_bgr*wrk_face_mask_area_a, dst_face_bgr*wrk_face_mask_area_a)

                        if cfg.mode == 'seamless-hist-match':
                            out_face_bgr = imagelib.color_hist_match(out_face_bgr, dst_face_bgr, cfg.hist_match_threshold)

                        if cfg_mp != 0:
                            k_size = int(frame_info.motion_power*cfg_mp)
                            if k_size >= 1:
                                k_size = np.clip (k_size+1, 2, 50)
                                if cfg.super_resolution_power != 0:
                                    k_size *= 2
                                out_face_bgr = imagelib.LinearMotionBlur (out_face_bgr, k_size , frame_info.motion_deg)

                        if cfg.blursharpen_amount != 0:
                            out_face_bgr = imagelib.blursharpen ( out_face_bgr, cfg.sharpen_mode, 3, cfg.blursharpen_amount)

                        if cfg.image_denoise_power != 0:
                            n = cfg.image_denoise_power
                            while n > 0:
                                img_bgr_denoised = cv2.medianBlur(img_bgr, 5)
                                if int(n / 100) != 0:
                                    img_bgr = img_bgr_denoised
                                else:
                                    pass_power = (n % 100) / 100.0
                                    img_bgr = img_bgr*(1.0-pass_power)+img_bgr_denoised*pass_power
                                n = max(n-10,0)

                        if cfg.bicubic_degrade_power != 0:
                            p = 1.0 - cfg.bicubic_degrade_power / 101.0
                            img_bgr_downscaled = cv2.resize (img_bgr, ( int(img_size[0]*p), int(img_size[1]*p ) ), interpolation=cv2.INTER_CUBIC)
                            img_bgr = cv2.resize (img_bgr_downscaled, img_size, interpolation=cv2.INTER_CUBIC)

                        new_out = cv2.warpAffine( out_face_bgr, face_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT )

                        out_img =  np.clip( img_bgr*(1-img_face_mask_a) + (new_out*img_face_mask_a) , 0, 1.0 )

                    if cfg.color_degrade_power != 0:
                        out_img_reduced = imagelib.reduce_colors(out_img, 256)
                        if cfg.color_degrade_power == 100:
                            out_img = out_img_reduced
                        else:
                            alpha = cfg.color_degrade_power / 100.0
                            out_img = (out_img*(1.0-alpha) + out_img_reduced*alpha)
            out_merging_mask_a = img_face_mask_a

    if out_img is None:
    
        out_img = img_bgr.copy()
        
    if cfg.show_mode == 3:
        
        g = dflimg.get_source_rect()
        
        a,b = 30, int(30*(g[3]-g[1])/(g[2]-g[0]))
        
        out = out_img[g[1]-int(1.5*a):g[3]+a,g[0]-b:g[2]+b]
        
    if  cfg.show_mode == 4:
    
        
        g = dflimg.get_source_rect()
        
        a,b = 30, int(30*(g[3]-g[1])/(g[2]-g[0]))
        
        
        
        out = img_bgr[g[1]-int(1.5*a):g[3]+a,g[0]-b:g[2]+b]
        
    if cfg.show_mode == 2:
    
        out = img_bgr
        
        
    if cfg.show_mode == 1:
    
        out = out_img
        
    return out
    
    
    