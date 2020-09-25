import sys
sys.path.append('/content/DeepFaceLab')
import multiprocessing
import os
import pickle
import traceback
from pathlib import Path

import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from merger import MergerConfig

import random
import sys
import traceback

import cv2
import numpy as np
import settings
from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor

is_windows = sys.platform[0:3] == 'win'
xseg_input_size = 256

import sys
import traceback

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor

is_windows = sys.platform[0:3] == 'win'
xseg_input_size = 256

def MergeMaskedFace (predictor_func, predictor_input_shape,
                     face_enhancer_func,
                     xseg_256_extract_func,
                     cfg, frame_info, img_bgr_uint8, img_bgr, img_face_landmarks):
                     
    data = [img_bgr, predictor_input_shape, frame_info, img_face_landmarks]

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

    dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_bgr      = np.clip(dst_face_bgr, 0, 1)

    dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    predictor_input_bgr      = cv2.resize (dst_face_bgr, (input_size,input_size) )

    predicted = predictor_func (predictor_input_bgr)
    prd_face_bgr          = np.clip (predicted[0], 0, 1.0)
    data.append(prd_face_bgr)
    prd_face_mask_a_0     = np.clip (predicted[1], 0, 1.0)
    data.append(prd_face_mask_a_0)
    prd_face_dst_mask_a_0 = np.clip (predicted[2], 0, 1.0)
    data.append(prd_face_dst_mask_a_0)

    if cfg.super_resolution_power != 0:
        prd_face_bgr_enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
        mod = cfg.super_resolution_power / 100.0
        prd_face_bgr = cv2.resize(prd_face_bgr, (output_size,output_size))*(1.0-mod) + prd_face_bgr_enhanced*mod
        prd_face_bgr = np.clip(prd_face_bgr, 0, 1)

    if cfg.super_resolution_power != 0:
        prd_face_mask_a_0     = cv2.resize (prd_face_mask_a_0,      (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        prd_face_dst_mask_a_0 = cv2.resize (prd_face_dst_mask_a_0,  (output_size, output_size), interpolation=cv2.INTER_CUBIC)

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
        if cfg.mask_mode == 6 or cfg.mask_mode == 8 or cfg.mask_mode == 9:
            # obtain XSeg-prd
            prd_face_xseg_bgr = cv2.resize (prd_face_bgr, (xseg_input_size,)*2, interpolation=cv2.INTER_CUBIC)
            prd_face_xseg_mask = xseg_256_extract_func(prd_face_xseg_bgr)
            X_prd_face_mask_a_0 = cv2.resize ( prd_face_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        if cfg.mask_mode >= 7 and cfg.mask_mode <= 9:
            # obtain XSeg-dst
            xseg_mat            = LandmarksProcessor.get_transform_mat (img_face_landmarks, xseg_input_size, face_type=cfg.face_type)
            dst_face_xseg_bgr   = cv2.warpAffine(img_bgr, xseg_mat, (xseg_input_size,)*2, flags=cv2.INTER_CUBIC )
            dst_face_xseg_mask  = xseg_256_extract_func(dst_face_xseg_bgr)
            X_dst_face_mask_a_0 = cv2.resize (dst_face_xseg_mask, (output_size,output_size), interpolation=cv2.INTER_CUBIC)

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
    data.append(img_bgr_uint8)
    if wrk_face_mask_a_0.shape[0] != output_size:
        wrk_face_mask_a_0 = cv2.resize (wrk_face_mask_a_0, (output_size,output_size), interpolation=cv2.INTER_CUBIC)

    wrk_face_mask_a = wrk_face_mask_a_0[...,None]

    out_img = None
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

                out_img = img_bgr*(1-img_face_mask_a) + (out_img*img_face_mask_a)

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
        
    return out_img, out_merging_mask_a, data


def MergeMasked (predictor_func,
                 predictor_input_shape,
                 face_enhancer_func,
                 xseg_256_extract_func,
                 cfg,
                 frame_info):
    img_bgr_uint8 = cv2_imread(frame_info.filepath)
    img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 3)
    img_bgr = img_bgr_uint8.astype(np.float32) / 255.0

    outs = []
    for face_num, img_landmarks in enumerate( frame_info.landmarks_list ):
        out_img, out_img_merging_mask, data = MergeMaskedFace (predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, cfg, frame_info, img_bgr_uint8, img_bgr, img_landmarks)
        np.array(data).dump('/tmp/data_'+str(frame_info.filepath).split('/')[-1].split('.')[0]+'_.npy')
        outs += [ (out_img, out_img_merging_mask) ]

    #Combining multiple face outputs
    final_img = None
    final_mask = None
    for img, merging_mask in outs:
        h,w,c = img.shape

        if final_img is None:
            final_img = img
            final_mask = merging_mask
        else:
            final_img = final_img*(1-merging_mask) + img*merging_mask
            final_mask = np.clip (final_mask + merging_mask, 0, 1 )

    final_img = np.concatenate ( [final_img, final_mask], -1)

    return (final_img*255).astype(np.uint8)







from core.leras import nn

nn.initialize_main_env()
import math
import traceback
from pathlib import Path

import numpy as np
import numpy.linalg as npla

import samplelib
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import MPClassFuncOnDemand, MPFunc
from core.leras import nn
from DFLIMG import DFLIMG
from facelib import FaceEnhancer, FaceType, LandmarksProcessor, XSegNet
from merger import FrameInfo, MergerConfig

def main (model_class_name=None,
          saved_models_path=None,
          training_data_src_path=None,
          force_model_name=None,
          input_path=None,
          output_path=None,
          output_mask_path=None,
          aligned_path=None,
          force_gpu_idxs=None,
          cpu_only=None):
    io.log_info ("Running merger.\r\n")

    try:
        if not input_path.exists():
            io.log_err('Input directory not found. Please ensure it exists.')
            return

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        if not output_mask_path.exists():
            output_mask_path.mkdir(parents=True, exist_ok=True)

        if not saved_models_path.exists():
            io.log_err('Model directory not found. Please ensure it exists.')
            return

        # Initialize model
        import models
        model = models.import_model(model_class_name)(is_training=False,
                                                      saved_models_path=saved_models_path,
                                                      force_gpu_idxs=force_gpu_idxs,
                                                      cpu_only=cpu_only)

        predictor_func, predictor_input_shape, cfg = model.get_MergerConfig()

        # Preparing MP functions
        predictor_func = MPFunc(predictor_func)

        run_on_cpu = len(nn.getCurrentDeviceConfig().devices) == 0
        xseg_256_extract_func = MPClassFuncOnDemand(XSegNet, 'extract',
                                                    name='XSeg',
                                                    resolution=256,
                                                    weights_file_root=saved_models_path,
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        face_enhancer_func = MPClassFuncOnDemand(FaceEnhancer, 'enhance',
                                                    place_model_on_cpu=True,
                                                    run_on_cpu=run_on_cpu)

        is_interactive = io.input_bool ("Use interactive merger?", True) if not io.is_colab() else False

        if not is_interactive:
            cfg.ask_settings()

        input_path_image_paths = pathex.get_image_paths(input_path)

        if cfg.type == MergerConfig.TYPE_MASKED:
            if not aligned_path.exists():
                io.log_err('Aligned directory not found. Please ensure it exists.')
                return

            packed_samples = None
            try:
                packed_samples = samplelib.PackedFaceset.load(aligned_path)
            except:
                io.log_err(f"Error occured while loading samplelib.PackedFaceset.load {str(aligned_path)}, {traceback.format_exc()}")


            if packed_samples is not None:
                io.log_info ("Using packed faceset.")
                def generator():
                    for sample in io.progress_bar_generator( packed_samples, "Collecting alignments"):
                        filepath = Path(sample.filename)
                        yield filepath, DFLIMG.load(filepath, loader_func=lambda x: sample.read_raw_file()  )
            else:
                def generator():
                    for filepath in io.progress_bar_generator( pathex.get_image_paths(aligned_path), "Collecting alignments"):
                        filepath = Path(filepath)
                        yield filepath, DFLIMG.load(filepath)

            alignments = {}
            multiple_faces_detected = False

            for filepath, dflimg in generator():
                if dflimg is None or not dflimg.has_data():
                    io.log_err (f"{filepath.name} is not a dfl image file")
                    continue

                source_filename = dflimg.get_source_filename()
                if source_filename is None:
                    continue

                source_filepath = Path(source_filename)
                source_filename_stem = source_filepath.stem

                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments_ar = alignments[ source_filename_stem ]
                alignments_ar.append ( (dflimg.get_source_landmarks(), filepath, source_filepath ) )

                if len(alignments_ar) > 1:
                    multiple_faces_detected = True

            if multiple_faces_detected:
                io.log_info ("")
                io.log_info ("Warning: multiple faces detected. Only one alignment file should refer one source file.")
                io.log_info ("")

            for a_key in list(alignments.keys()):
                a_ar = alignments[a_key]
                if len(a_ar) > 1:
                    for _, filepath, source_filepath in a_ar:
                        io.log_info (f"alignment {filepath.name} refers to {source_filepath.name} ")
                    io.log_info ("")

                alignments[a_key] = [ a[0] for a in a_ar]

            if multiple_faces_detected:
                io.log_info ("It is strongly recommended to process the faces separatelly.")
                io.log_info ("Use 'recover original filename' to determine the exact duplicates.")
                io.log_info ("")

            frames = [ InteractiveMergerSubprocessor.Frame( frame_info=FrameInfo(filepath=Path(p),
                                                                     landmarks_list=alignments.get(Path(p).stem, None)
                                                                    )
                                              )
                       for p in random.sample(input_path_image_paths, 20) ]

            if multiple_faces_detected:
                io.log_info ("Warning: multiple faces detected. Motion blur will not be used.")
                io.log_info ("")
            else:
                s = 256
                local_pts = [ (s//2-1, s//2-1), (s//2-1,0) ] #center+up
                frames_len = len(frames)
                for i in io.progress_bar_generator( range(len(frames)) , "Computing motion vectors"):
                    fi_prev = frames[max(0, i-1)].frame_info
                    fi      = frames[i].frame_info
                    fi_next = frames[min(i+1, frames_len-1)].frame_info
                    if len(fi_prev.landmarks_list) == 0 or \
                       len(fi.landmarks_list) == 0 or \
                       len(fi_next.landmarks_list) == 0:
                            continue

                    mat_prev = LandmarksProcessor.get_transform_mat ( fi_prev.landmarks_list[0], s, face_type=FaceType.FULL)
                    mat      = LandmarksProcessor.get_transform_mat ( fi.landmarks_list[0]     , s, face_type=FaceType.FULL)
                    mat_next = LandmarksProcessor.get_transform_mat ( fi_next.landmarks_list[0], s, face_type=FaceType.FULL)

                    pts_prev = LandmarksProcessor.transform_points (local_pts, mat_prev, True)
                    pts      = LandmarksProcessor.transform_points (local_pts, mat, True)
                    pts_next = LandmarksProcessor.transform_points (local_pts, mat_next, True)

                    prev_vector = pts[0]-pts_prev[0]
                    next_vector = pts_next[0]-pts[0]

                    motion_vector = pts_next[0] - pts_prev[0]
                    fi.motion_power = npla.norm(motion_vector)

                    motion_vector = motion_vector / fi.motion_power if fi.motion_power != 0 else np.array([0,0],dtype=np.float32)

                    fi.motion_deg = -math.atan2(motion_vector[1],motion_vector[0])*180 / math.pi


        if len(frames) == 0:
            io.log_info ("No frames to merge in input_dir.")
        else:
            if False:
                pass
            else:
                InteractiveMergerSubprocessor (
                            is_interactive         = is_interactive,
                            merger_session_filepath = model.get_strpath_storage_for_file('merger_session.dat'),
                            predictor_func         = predictor_func,
                            predictor_input_shape  = predictor_input_shape,
                            face_enhancer_func     = face_enhancer_func,
                            xseg_256_extract_func = xseg_256_extract_func,
                            merger_config          = cfg,
                            frames                 = frames,
                            frames_root_path       = input_path,
                            output_path            = output_path,
                            output_mask_path       = output_mask_path,
                            model_iter             = model.get_iter()
                        ).run()

        model.finalize()

    except Exception as e:
        print ( traceback.format_exc() )




MERGER_DEBUG = False
class InteractiveMergerSubprocessor(Subprocessor):

    class Frame(object):
        def __init__(self, prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None):
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = None
            self.output_mask_filepath = None

            self.idx = None
            self.cfg = None
            self.is_done = False
            self.is_processing = False
            self.is_shown = False
            self.image = None

    class ProcessingFrame(object):
        def __init__(self, idx=None,
                           cfg=None,
                           prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None,
                           output_filepath=None,
                           output_mask_filepath=None,
                           need_return_image = False):
            self.idx = idx
            self.cfg = cfg
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = output_filepath
            self.output_mask_filepath = output_mask_filepath

            self.need_return_image = need_return_image
            if self.need_return_image:
                self.image = None

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.log_info ('Running on %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.predictor_func = client_dict['predictor_func']
            self.predictor_input_shape = client_dict['predictor_input_shape']
            self.face_enhancer_func = client_dict['face_enhancer_func']
            self.xseg_256_extract_func = client_dict['xseg_256_extract_func']


            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            return None

        #override
        def process_data(self, pf): #pf=ProcessingFrame
            cfg = pf.cfg.copy()

            frame_info = pf.frame_info
            filepath = frame_info.filepath

            if len(frame_info.landmarks_list) == 0:
                
                if cfg.mode == 'raw-predict':        
                    h,w,c = self.predictor_input_shape
                    img_bgr = np.zeros( (h,w,3), dtype=np.uint8)
                    img_mask = np.zeros( (h,w,1), dtype=np.uint8)               
                else:                
                    self.log_info (f'no faces found for {filepath.name}, copying without faces')
                    img_bgr = cv2_imread(filepath)
                    imagelib.normalize_channels(img_bgr, 3)                    
                    h,w,c = img_bgr.shape
                    img_mask = np.zeros( (h,w,1), dtype=img_bgr.dtype)
                    
                cv2_imwrite (pf.output_filepath, img_bgr)
                cv2_imwrite (pf.output_mask_filepath, img_mask)

                if pf.need_return_image:
                    pf.image = np.concatenate ([img_bgr, img_mask], axis=-1)

            else:
                if cfg.type == MergerConfig.TYPE_MASKED:
                    try:
                        final_img = MergeMasked(self.predictor_func, self.predictor_input_shape,
                                                 face_enhancer_func=self.face_enhancer_func,
                                                 xseg_256_extract_func=self.xseg_256_extract_func,
                                                 cfg=cfg,
                                                 frame_info=frame_info)


                       

                        #print('done')
                    except Exception as e:
                        e_str = traceback.format_exc()
                        if 'MemoryError' in e_str:
                            raise Subprocessor.SilenceException
                        else:
                            raise Exception( f'Error while merging file [{filepath}]: {e_str}' )

                elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
                    final_img = MergeFaceAvatar (self.predictor_func, self.predictor_input_shape,
                                                   cfg, pf.prev_temporal_frame_infos,
                                                        pf.frame_info,
                                                        pf.next_temporal_frame_infos )

                cv2_imwrite (pf.output_filepath,      final_img[...,0:3] )
                cv2_imwrite (pf.output_mask_filepath, final_img[...,3:4] )

             

            return pf

        #overridable
        def get_data_name (self, pf):
            #return string identificator of your data
            return pf.frame_info.filepath




    #override
    def __init__(self, is_interactive, merger_session_filepath, predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, merger_config, frames, frames_root_path, output_path, output_mask_path, model_iter):
        if len (frames) == 0:
            raise ValueError ("len (frames) == 0")

        super().__init__('Merger', InteractiveMergerSubprocessor.Cli, io_loop_sleep_time=0.001)

        self.is_interactive = is_interactive
        self.merger_session_filepath = Path(merger_session_filepath)
        self.merger_config = merger_config

        self.predictor_func = predictor_func
        self.predictor_input_shape = predictor_input_shape

        self.face_enhancer_func = face_enhancer_func
        self.xseg_256_extract_func = xseg_256_extract_func

        self.frames_root_path = frames_root_path
        self.output_path = output_path
        self.output_mask_path = output_mask_path
        self.model_iter = model_iter

        self.prefetch_frame_count = self.process_count = multiprocessing.cpu_count()

        session_data = None
        if self.is_interactive and self.merger_session_filepath.exists():
            io.input_skip_pending()
            if io.input_bool ("Use saved session?", True):
                try:
                    with open( str(self.merger_session_filepath), "rb") as f:
                        session_data = pickle.loads(f.read())

                except Exception as e:
                    pass

        rewind_to_frame_idx = None
        self.frames = frames
        self.frames_idxs = [ *range(len(self.frames)) ]
        self.frames_done_idxs = []

        if self.is_interactive and session_data is not None:
            # Loaded session data, check it
            s_frames = session_data.get('frames', None)
            s_frames_idxs = session_data.get('frames_idxs', None)
            s_frames_done_idxs = session_data.get('frames_done_idxs', None)
            s_model_iter = session_data.get('model_iter', None)

            frames_equal = (s_frames is not None) and \
                           (s_frames_idxs is not None) and \
                           (s_frames_done_idxs is not None) and \
                           (s_model_iter is not None) and \
                           (len(frames) == len(s_frames)) # frames count must match

            if frames_equal:
                for i in range(len(frames)):
                    frame = frames[i]
                    s_frame = s_frames[i]
                    # frames filenames must match
                    if frame.frame_info.filepath.name != s_frame.frame_info.filepath.name:
                        frames_equal = False
                    if not frames_equal:
                        break

            if frames_equal:
                io.log_info ('Using saved session from ' + '/'.join (self.merger_session_filepath.parts[-2:]) )

                for frame in s_frames:
                    if frame.cfg is not None:
                        # recreate MergerConfig class using constructor with get_config() as dict params
                        # so if any new param will be added, old merger session will work properly
                        frame.cfg = frame.cfg.__class__( **frame.cfg.get_config() )

                self.frames = s_frames
                self.frames_idxs = s_frames_idxs
                self.frames_done_idxs = s_frames_done_idxs

                if self.model_iter != s_model_iter:
                    # model was more trained, recompute all frames
                    rewind_to_frame_idx = -1
                    for frame in self.frames:
                        frame.is_done = False
                elif len(self.frames_idxs) == 0:
                    # all frames are done?
                    rewind_to_frame_idx = -1

                if len(self.frames_idxs) != 0:
                    cur_frame = self.frames[self.frames_idxs[0]]
                    cur_frame.is_shown = False

            if not frames_equal:
                session_data = None

        if session_data is None:
            for filename in pathex.get_image_paths(self.output_path): #remove all images in output_path
                Path(filename).unlink()

            for filename in pathex.get_image_paths(self.output_mask_path): #remove all images in output_mask_path
                Path(filename).unlink()


            frames[0].cfg = self.merger_config.copy()

        for i in range( len(self.frames) ):
            frame = self.frames[i]
            frame.idx = i
            frame.output_filepath      = self.output_path      / ( frame.frame_info.filepath.stem + '.png' )
            frame.output_mask_filepath = self.output_mask_path / ( frame.frame_info.filepath.stem + '.png' )

            if not frame.output_filepath.exists() or \
               not frame.output_mask_filepath.exists():
                # if some frame does not exist, recompute and rewind
                frame.is_done = False
                frame.is_shown = False

                if rewind_to_frame_idx is None:
                    rewind_to_frame_idx = i-1
                else:
                    rewind_to_frame_idx = min(rewind_to_frame_idx, i-1)

        if rewind_to_frame_idx is not None:
            while len(self.frames_done_idxs) > 0:
                if self.frames_done_idxs[-1] > rewind_to_frame_idx:
                    prev_frame = self.frames[self.frames_done_idxs.pop()]
                    self.frames_idxs.insert(0, prev_frame.idx)
                else:
                    break
    #override
    def process_info_generator(self):
        r = [0] if MERGER_DEBUG else range(self.process_count)

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'predictor_func': self.predictor_func,
                                      'predictor_input_shape' : self.predictor_input_shape,
                                      'face_enhancer_func': self.face_enhancer_func,
                                      'xseg_256_extract_func' : self.xseg_256_extract_func,
                                      'stdin_fd': sys.stdin.fileno() if MERGER_DEBUG else None
                                      }

    #overridable optional
    def on_clients_initialized(self):
        io.progress_bar ("Merging", len(self.frames_idxs)+len(self.frames_done_idxs), initial=len(self.frames_done_idxs) )

        self.process_remain_frames = not self.is_interactive
        self.is_interactive_quitting = not self.is_interactive

        if self.is_interactive:
            help_images = {
                    MergerConfig.TYPE_MASKED :      cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_masked.jpg') ),
                    MergerConfig.TYPE_FACE_AVATAR : cv2_imread ( str(Path(__file__).parent / 'gfx' / 'help_merger_face_avatar.jpg') ),
                }

            self.main_screen = Screen(initial_scale_to_width=1368, image=None, waiting_icon=True)
            self.help_screen = Screen(initial_scale_to_height=768, image=help_images[self.merger_config.type], waiting_icon=False)
            self.screen_manager = ScreenManager( "Merger", [self.main_screen, self.help_screen], capture_keys=True )
            self.screen_manager.set_current (self.help_screen)
            self.screen_manager.show_current()

            self.masked_keys_funcs = {
                    '`' : lambda cfg,shift_pressed: cfg.set_mode(0),
                    '1' : lambda cfg,shift_pressed: cfg.set_mode(1),
                    '2' : lambda cfg,shift_pressed: cfg.set_mode(2),
                    '3' : lambda cfg,shift_pressed: cfg.set_mode(3),
                    '4' : lambda cfg,shift_pressed: cfg.set_mode(4),
                    '5' : lambda cfg,shift_pressed: cfg.set_mode(5),
                    '6' : lambda cfg,shift_pressed: cfg.set_mode(6),
                    'q' : lambda cfg,shift_pressed: cfg.add_hist_match_threshold(1 if not shift_pressed else 5),
                    'a' : lambda cfg,shift_pressed: cfg.add_hist_match_threshold(-1 if not shift_pressed else -5),
                    'w' : lambda cfg,shift_pressed: cfg.add_erode_mask_modifier(1 if not shift_pressed else 5),
                    's' : lambda cfg,shift_pressed: cfg.add_erode_mask_modifier(-1 if not shift_pressed else -5),
                    'e' : lambda cfg,shift_pressed: cfg.add_blur_mask_modifier(1 if not shift_pressed else 5),
                    'd' : lambda cfg,shift_pressed: cfg.add_blur_mask_modifier(-1 if not shift_pressed else -5),
                    'r' : lambda cfg,shift_pressed: cfg.add_motion_blur_power(1 if not shift_pressed else 5),
                    'f' : lambda cfg,shift_pressed: cfg.add_motion_blur_power(-1 if not shift_pressed else -5),
                    't' : lambda cfg,shift_pressed: cfg.add_super_resolution_power(1 if not shift_pressed else 5),
                    'g' : lambda cfg,shift_pressed: cfg.add_super_resolution_power(-1 if not shift_pressed else -5),
                    'y' : lambda cfg,shift_pressed: cfg.add_blursharpen_amount(1 if not shift_pressed else 5),
                    'h' : lambda cfg,shift_pressed: cfg.add_blursharpen_amount(-1 if not shift_pressed else -5),
                    'u' : lambda cfg,shift_pressed: cfg.add_output_face_scale(1 if not shift_pressed else 5),
                    'j' : lambda cfg,shift_pressed: cfg.add_output_face_scale(-1 if not shift_pressed else -5),
                    'i' : lambda cfg,shift_pressed: cfg.add_image_denoise_power(1 if not shift_pressed else 5),
                    'k' : lambda cfg,shift_pressed: cfg.add_image_denoise_power(-1 if not shift_pressed else -5),
                    'o' : lambda cfg,shift_pressed: cfg.add_bicubic_degrade_power(1 if not shift_pressed else 5),
                    'l' : lambda cfg,shift_pressed: cfg.add_bicubic_degrade_power(-1 if not shift_pressed else -5),
                    'p' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(1 if not shift_pressed else 5),
                    ';' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(-1),
                    ':' : lambda cfg,shift_pressed: cfg.add_color_degrade_power(-5),
                    'z' : lambda cfg,shift_pressed: cfg.toggle_masked_hist_match(),
                    'x' : lambda cfg,shift_pressed: cfg.toggle_mask_mode(),
                    'c' : lambda cfg,shift_pressed: cfg.toggle_color_transfer_mode(),
                    'n' : lambda cfg,shift_pressed: cfg.toggle_sharpen_mode(),
                    }
            self.masked_keys = list(self.masked_keys_funcs.keys())

    #overridable optional
    def on_clients_finalized(self):
        io.progress_bar_close()

        if self.is_interactive:
            self.screen_manager.finalize()

            for frame in self.frames:
                frame.output_filepath = None
                frame.output_mask_filepath = None
                frame.image = None

            session_data = {
                'frames': self.frames,
                'frames_idxs': self.frames_idxs,
                'frames_done_idxs': self.frames_done_idxs,
                'model_iter' : self.model_iter,
            }
            self.merger_session_filepath.write_bytes( pickle.dumps(session_data) )

            io.log_info ("Session is saved to " + '/'.join (self.merger_session_filepath.parts[-2:]) )

    #override
    def on_tick(self):
        io.process_messages()

        go_prev_frame = False
        go_first_frame = False
        go_prev_frame_overriding_cfg = False
        go_first_frame_overriding_cfg = False

        go_next_frame = self.process_remain_frames
        go_next_frame_overriding_cfg = False
        go_last_frame_overriding_cfg = False

        cur_frame = None
        if len(self.frames_idxs) != 0:
            cur_frame = self.frames[self.frames_idxs[0]]

        if self.is_interactive:

            screen_image = None if self.process_remain_frames else \
                                   self.main_screen.get_image()

            self.main_screen.set_waiting_icon( self.process_remain_frames or \
                                               self.is_interactive_quitting )

            if cur_frame is not None and not self.is_interactive_quitting:

                if not self.process_remain_frames:
                    if cur_frame.is_done:
                        if not cur_frame.is_shown:
                            if cur_frame.image is None:
                                image      = cv2_imread (cur_frame.output_filepath, verbose=False)
                                image_mask = cv2_imread (cur_frame.output_mask_filepath, verbose=False)
                                if image is None or image_mask is None:
                                    # unable to read? recompute then
                                    cur_frame.is_done = False
                                else:
                                    image = imagelib.normalize_channels(image, 3)
                                    image_mask = imagelib.normalize_channels(image_mask, 1)
                                    cur_frame.image = np.concatenate([image, image_mask], -1)

                            if cur_frame.is_done:
                                io.log_info (cur_frame.cfg.to_string( cur_frame.frame_info.filepath.name) )
                                cur_frame.is_shown = True
                                screen_image = cur_frame.image
                    else:
                        self.main_screen.set_waiting_icon(True)

            self.main_screen.set_image(screen_image)
            self.screen_manager.show_current()

            key_events = self.screen_manager.get_key_events()
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

            if key == 9: #tab
                self.screen_manager.switch_screens()
            else:
                if key == 27: #esc
                    self.is_interactive_quitting = True
                elif self.screen_manager.get_current() is self.main_screen:

                    if self.merger_config.type == MergerConfig.TYPE_MASKED and chr_key in self.masked_keys:
                        self.process_remain_frames = False

                        if cur_frame is not None:
                            cfg = cur_frame.cfg
                            prev_cfg = cfg.copy()

                            if cfg.type == MergerConfig.TYPE_MASKED:
                                self.masked_keys_funcs[chr_key](cfg, shift_pressed)

                            if prev_cfg != cfg:
                                io.log_info ( cfg.to_string(cur_frame.frame_info.filepath.name) )
                                cur_frame.is_done = False
                                cur_frame.is_shown = False
                    else:

                        if chr_key == ',' or chr_key == 'm':
                            self.process_remain_frames = False
                            go_prev_frame = True

                            if chr_key == ',':
                                if shift_pressed:
                                    go_first_frame = True

                            elif chr_key == 'm':
                                if not shift_pressed:
                                    go_prev_frame_overriding_cfg = True
                                else:
                                    go_first_frame_overriding_cfg = True

                        elif chr_key == '.' or chr_key == '/':
                            self.process_remain_frames = False
                            go_next_frame = True

                            if chr_key == '.':
                                if shift_pressed:
                                    self.process_remain_frames = not self.process_remain_frames

                            elif chr_key == '/':
                                if not shift_pressed:
                                    go_next_frame_overriding_cfg = True
                                else:
                                    go_last_frame_overriding_cfg = True

                        elif chr_key == '-':
                            self.screen_manager.get_current().diff_scale(-0.1)
                        elif chr_key == '=':
                            self.screen_manager.get_current().diff_scale(0.1)
                        elif chr_key == 'v':
                            self.screen_manager.get_current().toggle_show_checker_board()

        if go_prev_frame:
            if cur_frame is None or cur_frame.is_done:
                if cur_frame is not None:
                    cur_frame.image = None

                while True:
                    if len(self.frames_done_idxs) > 0:
                        prev_frame = self.frames[self.frames_done_idxs.pop()]
                        self.frames_idxs.insert(0, prev_frame.idx)
                        prev_frame.is_shown = False
                        io.progress_bar_inc(-1)

                        if cur_frame is not None and (go_prev_frame_overriding_cfg or go_first_frame_overriding_cfg):
   
                            if prev_frame.cfg != cur_frame.cfg:
                                prev_frame.cfg = cur_frame.cfg.copy()
                                prev_frame.is_done = False

                        cur_frame = prev_frame

                    if go_first_frame_overriding_cfg or go_first_frame:
                        if len(self.frames_done_idxs) > 0:
                            continue
                    break

        elif go_next_frame:
            if cur_frame is not None and cur_frame.is_done:
                cur_frame.image = None
                cur_frame.is_shown = True
                self.frames_done_idxs.append(cur_frame.idx)
                self.frames_idxs.pop(0)
                io.progress_bar_inc(1)

                f = self.frames

                if len(self.frames_idxs) != 0:
                    next_frame = f[ self.frames_idxs[0] ]
                    next_frame.is_shown = False

                    if go_next_frame_overriding_cfg or go_last_frame_overriding_cfg:

                        if go_next_frame_overriding_cfg:
                            to_frames = next_frame.idx+1
                        else:
                            to_frames = len(f)

                        for i in range( next_frame.idx, to_frames ):
                            f[i].cfg = None

                    for i in range( min(len(self.frames_idxs), self.prefetch_frame_count) ):
                        frame = f[ self.frames_idxs[i] ]
                        if frame.cfg is None:
                            if i == 0:
                                frame.cfg = cur_frame.cfg.copy()
                            else:
                                frame.cfg = f[ self.frames_idxs[i-1] ].cfg.copy()

                            frame.is_done = False #initiate solve again
                            frame.is_shown = False

            if len(self.frames_idxs) == 0:
                self.process_remain_frames = False

        return (self.is_interactive and self.is_interactive_quitting) or \
               (not self.is_interactive and self.process_remain_frames == False)


    #override
    def on_data_return (self, host_dict, pf):
        frame = self.frames[pf.idx]
        frame.is_done = False
        frame.is_processing = False

    #override
    def on_result (self, host_dict, pf_sent, pf_result):
        frame = self.frames[pf_result.idx]
        frame.is_processing = False
        if frame.cfg == pf_result.cfg:
            frame.is_done = True
            frame.image = pf_result.image

    #override
    def get_data(self, host_dict):
        if self.is_interactive and self.is_interactive_quitting:
            return None

        for i in range ( min(len(self.frames_idxs), self.prefetch_frame_count) ):
            frame = self.frames[ self.frames_idxs[i] ]

            if not frame.is_done and not frame.is_processing and frame.cfg is not None:
                frame.is_processing = True
                return InteractiveMergerSubprocessor.ProcessingFrame(idx=frame.idx,
                                                           cfg=frame.cfg.copy(),
                                                           prev_temporal_frame_infos=frame.prev_temporal_frame_infos,
                                                           frame_info=frame.frame_info,
                                                           next_temporal_frame_infos=frame.next_temporal_frame_infos,
                                                           output_filepath=frame.output_filepath,
                                                           output_mask_filepath=frame.output_mask_filepath,
                                                           need_return_image=True )

        return None

    #override
    def get_result(self):
        return 0
main("SAEHD", 
     Path("workspace/model"),
     input_path = Path("workspace/preview"),
     output_path = Path("workspace/preview/merged"),
     output_mask_path =Path("workspace/preview/merged_mask"),
     aligned_path = Path("workspace/preview/aligned") )