import numpy as np

## all latency measurements were averaged over 10 runs using the Delphi codebase ## 

##################################### OFFLINE SECTION #####################################################
###########################################################################################################

off_client_compute_keygen     = 0.340    # seconds
off_client_write_key          = 33180308 # bytes

off_client_compute_he_encrypt = np.array([0.0441898, 0.083641 , 0.076235 , 0.0913766, 0.0758293, 0.0762482,
                                          0.077361 , 0.077221 , 0.0778429, 0.0778458, 0.0774584, 0.0780227,
                                          0.0378845, 0.0414266, 0.0422988, 0.0415863, 0.0419623, 0.0418264,
                                          0.041697 , 0.0411746, 0.0418523, 0.0413338, 0.0373184, 0.0395193,
                                          0.0401926, 0.0393085, 0.0394028, 0.0397033, 0.0392775, 0.0393418,
                                          0.039335 , 0.0061115]) # seconds

off_client_write_linear       = np.array([4719545, 9439082, 9439082, 9439082,
                                          9439082, 9439082, 9439082, 9439082,
                                          9439082, 9439082, 9439082, 9439082,
                                          4719545, 4719545, 4719545, 4719545,
                                          4719545, 4719545, 4719545, 4719545,
                                          4719545, 4719545, 4719545, 4719545,
                                          4719545, 4719545, 4719545, 4719545,
                                          4719545, 4719545, 4719545, 52440]) # bytes

off_client_compute_he_decrypt = np.array([0.0066864, 0.0065111, 0.0069805, 0.0064177, 0.006648 , 0.0067377,
                                          0.0072704, 0.006842 , 0.0072382, 0.0071379, 0.0068499, 0.0083139,
                                          0.0034627, 0.0036479, 0.0036391, 0.0036363, 0.0036548, 0.0036399,
                                          0.0035279, 0.0036604, 0.0036424, 0.0059437, 0.0030449, 0.0034824,
                                          0.0034158, 0.0032056, 0.0033947, 0.0028116, 0.0031826, 0.0034   ,
                                          0.0031257, 0.0041503]) # seconds

off_client_write_base_ot      = 4096      # bytes
off_client_write_ext_ot_setup = 203685888  # bytes


off_server_compute_he_eval    = np.array([0.4340473, 0.8534728, 0.8451628, 0.8462032,
                                          0.8482882, 0.8468683, 0.8542785, 0.8508684,
                                          0.8449283, 0.8450555, 0.8481074, 1.6846   ,
                                          0.9235332, 0.9335266, 0.9210749, 0.9209507,
                                          0.9220206, 0.9217837, 0.927072,  0.9211946,
                                          0.9218167, 1.8421,    2.3069,    2.2638   ,
                                          2.2497   , 2.1394,    2.2982,    2.3088   ,
                                          2.1093   , 2.1945,    2.2594,    
                                          0.0547067]) # seconds

off_server_write_linear       = np.array([1048794, 1048794, 1048794, 1048794,
                                          1048794, 1048794, 1048794, 1048794,
                                          1048794, 1048794, 1048794, 2097580,
                                          524401,  524401,  524401,  524401,
                                          524401,  524401,  524401,  524401,
                                          524401,  1048794, 524401,  524401,
                                          524401,  524401,  524401,  524401,
                                          524401,  524401,  524401,  
                                          524401]) # bytes

off_server_compute_garble     = 9.5205     # seconds
off_server_compute_encode     = 2.1203     # seconds
off_server_write_garbled_c    = 5174592080 # bytes
off_server_write_ext_ot_send  = 407371776  # bytes

###########################################################################################################
###########################################################################################################



##################################### ONLINE SECTION ######################################################
###########################################################################################################

on_client_write_linear   = np.array([27689, 147497, 147497, 147497, 147497, 147497, 147497, 147497,
                                     147497, 147497, 147497, 147497,  73769,  73769,  73769,  73769,
                                     73769,  73769,  73769,  73769,  73769,  73769,  36905,  36905,
                                     36905,  36905,  36905,  36905,  36905,  36905,  36905,  0, 
                                     9257]) # bytes

on_client_compute_relu   = np.array([0.3004942, 0.2918977, 0.2920599, 0.2889828, 0.2839677, 0.2917974,
                                     0.2926883, 0.2897318, 0.2913337, 0.3040986, 0.3028448, 0.1528664,
                                     0.1540935, 0.1531371, 0.1534293, 0.1533323, 0.1533467, 0.1490701,
                                     0.1490967, 0.1557279, 0.1514888, 0.0778148, 0.0798365, 0.0776657,
                                     0.0771422, 0.0773719, 0.0771831, 0.079022 , 0.0794517, 0.0773419,
                                     0.0769044, 0., 0.]) # seconds

on_server_write_relu     = np.array([19398664, 19398664, 19398664, 19398664, 19398664, 19398664,
                                     19398664, 19398664, 19398664, 19398664, 19398664,  9699336,
                                     9699336,  9699336,  9699336,  9699336,  9699336,  9699336,
                                     9699336,  9699336,  9699336,  4849672,  4849672,  4849672,
                                     4849672,  4849672,  4849672,  4849672,  4849672,  4849672,
                                     4849672, 0, 0.]) # bytes

on_server_compute_linear = np.array([6.12400e-02, 6.79000e-03, 7.17300e-03, 6.86400e-03, 7.14500e-03,
                                     7.77000e-03, 7.14100e-03, 7.34000e-03, 7.34200e-03, 7.31200e-03,
                                     7.35900e-03, 7.44500e-03, 7.17000e-03, 6.76100e-03, 6.83000e-03,
                                     6.78200e-03, 6.86300e-03, 6.95600e-03, 4.24200e-03, 6.78100e-03,
                                     6.89300e-03, 6.86400e-03, 7.88800e-03, 7.84200e-03, 8.17100e-03,
                                     8.20900e-03, 8.57300e-03, 8.80500e-03, 7.99300e-03, 8.23400e-03,
                                     7.19900e-03, 3.94390e-05, 1.52118e-04]) # seconds

on_server_write_pred     = 131 # bytes

###########################################################################################################
###########################################################################################################

