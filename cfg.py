cfg = {
    "batch_vector_num" : 2000,
    "batch_num" : 8,

    "d_hidden_mul" : 64,
    "l1_coeff" : 3e-4,
    "save_dir" : "./checkpoints/",

    "device" : "cuda:0",
    "seed" : 49,
    "rare_threshold": 10 ** -4.5,
    "resample_threshold": 10 ** -5,
    # "rare_threshold": 0.3,
    "do_resampling": False,
    "iters_to_log": 100,
    "iters_to_save": 3000
}

thresholds = [-3, -3, -2, -2.5, -2.5, -3.5, -3, -4, -4]
