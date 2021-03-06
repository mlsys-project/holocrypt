import numpy as np

## all latency measurements were averaged over 5 runs using the Delphi codebase ## 

##################################### OFFLINE SECTION #####################################################
###########################################################################################################

off_client_compute_keygen     = 0.340    # seconds
off_client_write_key          = 33180308 # bytes

off_client_compute_he_encrypt = np.array([0.086055 , 0.3543974, 0.3270482, 0.3265828, 0.3288516, 0.3298392,
                                          0.3302058, 0.3289602, 0.329609 , 0.331795 , 0.3299792, 0.3340208,
                                          0.1522108, 0.1546402, 0.1538434, 0.1553902, 0.1564748, 0.1538696,
                                          0.1539244, 0.1517034, 0.1573964, 0.155639 , 0.0735008, 0.0747708,
                                          0.0738456, 0.0739048, 0.0736066, 0.0731986, 0.073261 , 0.0736974,
                                          0.073572 , 0.0043562]) # seconds CHANGED

off_client_write_linear       = np.array([ 9439082, 37756304, 37756304, 37756304, 37756304, 37756304,
                                          37756304, 37756304, 37756304, 37756304, 37756304, 37756304,
                                          18878156, 18878156, 18878156, 18878156, 18878156, 18878156,
                                          18878156, 18878156, 18878156, 18878156,  9439082,  9439082,
                                           9439082,  9439082,  9439082,  9439082,  9439082,  9439082,
                                           9439082,   524401]) # bytes CHANGED

off_client_compute_he_decrypt = np.array([0.0124424, 0.0133748, 0.0125544, 0.0125676, 0.0125574, 0.012603 ,
                                          0.0125548, 0.0126028, 0.0125966, 0.0125348, 0.0124826, 0.0247022,
                                          0.0087074, 0.00899  , 0.0079684, 0.0088984, 0.0094192, 0.0090722,
                                          0.0089374, 0.0082528, 0.0089358, 0.012159 , 0.0067322, 0.006033 ,
                                          0.0057152, 0.006657 , 0.0060918, 0.005937 , 0.0059546, 0.0060308,
                                          0.006675 , 0.0041248]) # seconds CHANGED

off_client_write_base_ot      = 4096      # bytes CHANGED
off_client_write_ext_ot_setup = 814743552 # bytes CHANGED


off_server_compute_he_eval    = np.array([0.7997016, 3.1246   , 3.1402   , 3.1152   , 3.1294   , 3.121    ,
                                          3.1272   , 3.1114   , 3.1236   , 3.1028   , 3.114    , 6.5722   ,
                                          3.336    , 3.34     , 3.3434   , 3.3462   , 3.337    , 3.3344   ,
                                          3.3324   , 3.343    , 3.3366   , 6.6734   , 3.6402   , 3.6362   ,
                                          3.6372   , 3.6352   , 3.656    , 3.6342   , 3.6552   , 3.6356   ,
                                          3.6472   , 1.749    ]) # seconds CHANGED


off_server_write_linear       = np.array([4195152, 4195152, 4195152, 4195152, 4195152, 4195152, 4195152,
                                          4195152, 4195152, 4195152, 4195152, 8390296, 2097580, 2097580,
                                          2097580, 2097580, 2097580, 2097580, 2097580, 2097580, 2097580,
                                          4195152, 1048794, 1048794, 1048794, 1048794, 1048794, 1048794,
                                          1048794, 1048794, 1048794,  524401]) # bytes CHANGED

off_server_compute_garble     = 36.9232     # seconds CHANGED
off_server_compute_encode     = 8.878599    # seconds CHANGED
off_server_write_garbled_c    = 20698368320 # bytes CHANGED
off_server_write_ext_ot_send  = 1629487104  # bytes CHANGED

###########################################################################################################
###########################################################################################################



##################################### ONLINE SECTION ######################################################
###########################################################################################################

on_client_write_linear   = np.array([110633, 589865, 589865, 589865, 589865, 589865, 589865, 589865,
                                     589865, 589865, 589865, 589865, 294953, 294953, 294953, 294953,
                                     294953, 294953, 294953, 294953, 294953, 294953, 147497, 147497,
                                     147497, 147497, 147497, 147497, 147497, 147497, 147497,      0,
                                      36905]) # bytes CHANGED

on_client_compute_relu   = np.array([1.1638   , 1.1744   , 1.1594   , 1.1564   , 1.1672   , 1.2714   ,
                                     1.3148   , 1.3564   , 1.3844   , 1.4094   , 1.4118   , 0.6721838,
                                     0.6473926, 0.717527 , 0.7135034, 0.694175 , 0.695092 , 0.7131196,
                                     0.7366724, 0.7308796, 0.7372748, 0.3721364, 0.369593 , 0.3848076,
                                     0.3723588, 0.36842  , 0.3694872, 0.3698434, 0.3407292, 0.3537946,
                                     0.3709386, 0.       , 0.]) # seconds CHANGED

on_server_write_relu     = np.array([77594630., 77594630., 77594630., 77594630., 77594630., 77594630.,
                                     77594630., 77594630., 77594630., 77594630., 77594630., 38797320.,
                                     38797320., 38797320., 38797320., 38797320., 38797320., 38797320.,
                                     38797320., 38797320., 38797320., 19398664., 19398664., 19398664.,
                                     19398664., 19398664., 19398664., 19398664., 19398664., 19398664.,
                                     19398664.,        0., 0.]) # bytes CHANGED

on_server_compute_linear = np.array([0.0621554 , 0.0209662 , 0.0195872 , 0.020306  , 0.019071  ,
                                     0.0189206 , 0.0193864 , 0.020304  , 0.0202732 , 0.0206366 ,
                                     0.0197006 , 0.0198372 , 0.0079442 , 0.0082076 , 0.00998   ,
                                     0.0092086 , 0.0071744 , 0.0080324 , 0.0100902 , 0.009388  ,
                                     0.008148  , 0.0088562 , 0.0113578 , 0.0091276 , 0.0081334 ,
                                     0.0091    , 0.0093374 , 0.0077022 , 0.0069806 , 0.0103756 ,
                                     0.0099572 , 0.00015334, 0.0097322 ]) # seconds CHANGED

on_server_write_pred     = 1841 # bytes

###########################################################################################################
###########################################################################################################

