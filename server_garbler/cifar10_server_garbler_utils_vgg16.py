import numpy as np

## all latency measurements were averaged over 10 runs using the Delphi codebase ## 



##################################### OFFLINE SECTION #####################################################
###########################################################################################################

off_client_compute_keygen     = 0.340    # seconds CHANGED
off_client_write_key          = 33180308 # bytes CHANGED

off_client_compute_he_encrypt = np.array([0.0429106, 0.3474835, 0.0755515, 0.1522346, 0.0370924, 0.076652 ,
                                          0.0768156, 0.0369633, 0.0401408, 0.0399115, 0.0396316, 0.0400213,
                                          0.0398017, 0.0067634, 0.0067803, 0.0072326]) #seconds CHANGED

off_client_write_linear       = np.array([4719545, 37756304,  9439082, 18878156,  4719545,
                                          9439082,  9439082,  4719545,  4719545,  4719545,  4719545,
                                          4719545,  4719545,   524401,   524401,   524401]) # bytes CHANGED

off_client_compute_he_decrypt = np.array([0.012585 , 0.0128849, 0.0086443, 0.0075462, 0.0069662, 0.0069989,
                                          0.0066802, 0.0035374, 0.0035932, 0.0035681, 0.0034375, 0.0034413,
                                          0.0034411, 0.0035304, 0.0037   , 0.0035953]) # seconds CHANGED

off_client_write_base_ot      = 4096      # bytes #CHANGED
off_client_write_ext_ot_setup = 191299584 # bytes #CHANGED 



off_server_compute_he_eval    = np.array([ 1.6827  , 13.5379  ,  7.3134  , 14.6433  ,  7.9497  , 15.9169  ,
                                           15.8851  , 17.2357  , 17.2318  , 17.2279  , 18.5745  , 18.5912  ,
                                           18.5593  ,  3.811   , 38.6636  ,  0.108043]) # seconds CHANGED

off_server_write_linear       = np.array([4195152, 4195152, 2097580, 2097580, 1048794, 1048794, 1048794,
                                          524401,  524401,  524401,  524401,  524401,  524401,  524401,
                                          524401,  524401]) # bytes CHANGED

off_server_compute_garble     = 8.7896     # seconds CHANGED
off_server_compute_encode     = 2.295      # seconds CHANGED
off_server_write_garbled_c    = 4859920940 # bytes CHANGED
off_server_write_ext_ot_send  = 382599168  # bytes CHANGED

###########################################################################################################
###########################################################################################################



##################################### ONLINE SECTION ######################################################
###########################################################################################################

on_client_write_linear   = np.array([ 27689, 589865,      0, 147497, 294953,      0,  73769, 147497,
                                      147497,      0,  36905,  73769,  73769,      0,  18473,  18473,
                                      18473,      0,   4649,  36905,  36905]) # bytes CHANGED

on_client_compute_relu   = np.array([1.1166   , 1.5348   , 0, 0.5577935, 0.5597488, 0, 0.2914538, 0.2895338,
                                     0.2863851, 0, 0.1410602, 0.1451064, 0.1416594, 0, 0.0378434, 0.039374 ,
                                     0.0398178, 0, 0.0705095, 0.0721002, 0.]) #seconds CHANGED

on_server_write_relu     = np.array([77594630, 77594630, 0, 38797320, 38797320, 0, 19398664, 19398664,
                                     19398664,  0, 9699336,  9699336,  9699336,  0, 2424840,  2424840,
                                     2424840,  0, 4849672,  4849672, 0]) #bytes CHANGED

on_server_compute_linear = np.array([6.058630e-02, 2.142360e-02, 7.306386e-04, 1.172150e-02,
                                     1.409550e-02, 4.527827e-04, 1.607400e-02, 1.862850e-02,
                                     2.056620e-02, 1.730500e-04, 2.820050e-02, 4.454430e-02,
                                     3.435710e-02, 8.783580e-05, 3.892810e-02, 3.686150e-02,
                                     2.904150e-02, 2.405370e-05, 2.032750e-02, 1.433169e-01,
                                     9.526849e-04]) #seconds CHANGED

on_server_write_pred     = 131 # bytes CHANGED

###########################################################################################################
###########################################################################################################
