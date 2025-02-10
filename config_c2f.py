data_config = {
    "CASIA": {
        "image_directory": "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/cropped_images",
        "known_list_path":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/pkl/CASIA_known_list.pkl",
        "unknown_list_path":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/pkl/CASIA_unknown_list.pkl",
    },
    "IJBC":{
        "image_directory" : "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/images",
        "img_root_loose_crop" : "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/loose_crop",
        "ijbc_t_m": "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/meta/ijbc_face_tid_mid.txt",
        "ijbc_5pts":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/meta/ijbc_name_5pts_score.txt",
        "ijbc_gallery_1":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/meta/ijbc_1N_gallery_G1.csv",
        "ijbc_gallery_2":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/meta/ijbc_1N_gallery_G2.csv",
        "ijbc_probe":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/meta/ijbc_1N_probe_mixed.csv",
        "plk_file_root":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/IJBC/plk"
    },
    "VGGFACE": {
            "image_directory": "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/vggface2_mtcnn_160",
            "known_list_path":"/home/chen/python_workspace/face/OSFI-by-FineTuning-main/pkl/selected_vgg_subfolder_names.pkl",
        },
}

encoder_config = {
    "VGG19": "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/VGG19_CosFace.chkpt",
    "Res50": "/home/chen/python_workspace/face/OSFI-by-FineTuning-main/ResIR50_CosFace.chkpt",
}