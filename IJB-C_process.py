import os
import csv
import pandas as pd
from itertools import islice
from config_c2f import data_config
from tqdm import tqdm
import shutil
import pickle
import numpy as np
data_config = data_config['IJBC']

# The path to the meta file
ijbc_gallery_1 = data_config["ijbc_gallery_1"]
ijbc_gallery_2 = data_config["ijbc_gallery_2"]
ijbc_probe = data_config["ijbc_probe"]
ijbc_t_m = data_config["ijbc_t_m"]
ijbc_5pts = data_config["ijbc_5pts"]

# Still image identity id and filename mapping file
still_img_id_filename_pair_file = 'ijbc_img_id_filename.pkl'

# All identity ids and filename mapping files
id_filename_pair_file = 'ijbc_id_filename.pkl'

# Original image file location
img_root = data_config["img_root_loose_crop"]

# Location of the processed image
'''
    - images
        - subject_id 1
            - image_name 1
            - image_name 2
            - image_name 3
        - subject_id 2
            - image_name 4
            - image_name 5 
            - image_name 6
'''
processed_img_root = data_config["image_directory"]

id_filename_pair_img = {}
id_filename_pair = {}

t_m_current_position = 0
landmark_current_position = 0

def read_gallery(filename, file_t_m, file_landmark, img):
    global t_m_current_position
    global landmark_current_position
    df = pd.read_csv(filename)
    rows = df[["TEMPLATE_ID", "SUBJECT_ID"]]
    for index, row in tqdm(rows.iterrows()):
        template_id, subject_id = int(row[0]), int(row[1])
        file_t_m.seek(t_m_current_position, 0)
        file_landmark.seek(landmark_current_position, 0)
        current_line_t_m = file_t_m.readline()
        current_line_landmark = file_landmark.readline()
        file_name, t_id, _media = current_line_t_m.strip().split(' ')
        lmk = current_line_landmark.strip().split(' ')[1:-1]
        lmk = np.array([float(x) for x in lmk],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        if 'cropped_images' in img_root:
            file_name = os.path.join(img_root, str(subject_id), file_name)
        else:
            file_name = os.path.join(img_root, file_name)
        if subject_id not in id_filename_pair:
            id_filename_pair[subject_id] = [{"filename": file_name,"lmk": lmk}]
            if img:
                id_filename_pair_img[subject_id] = [{"filename": file_name,"lmk": lmk}]
        else:
            id_filename_pair[subject_id].append({"filename": file_name,"lmk": lmk})
            if img:
                id_filename_pair_img[subject_id].append({"filename": file_name,"lmk": lmk})
        t_m_current_position = file_t_m.tell()
        landmark_current_position = file_landmark.tell()

def get_img():
    global t_m_current_position
    global landmark_current_position
    with open(ijbc_t_m, 'r') as file_t_m:
        with open(ijbc_5pts, 'r') as file_landmark:
            read_gallery(ijbc_gallery_1, file_t_m, file_landmark, True)
            read_gallery(ijbc_gallery_2, file_t_m, file_landmark, True)

            t_m_gallery_position = t_m_current_position
            landmark_gallery_position = landmark_current_position
            df = pd.read_csv(ijbc_probe)
            # Filter out the lines that start with 'img'
            rows_with_img = df[df["FILENAME"].str.startswith('img')][["TEMPLATE_ID", "SUBJECT_ID", "SIGHTING_ID"]]
            for index, row in rows_with_img.iterrows():
                template_id, subject_id, media_id = row[0], row[1], row[2]
                Next = True
                while(Next):
                    file_t_m.seek(t_m_current_position, 0)
                    file_landmark.seek(landmark_current_position, 0)
                    current_line_t_m = file_t_m.readline()
                    current_line_landmark = file_landmark.readline()
                    file_name, _id, _media = current_line_t_m.strip().split(' ')
                    lmk = current_line_landmark.strip().split(' ')[1:-1]
                    lmk = np.array([float(x) for x in lmk],
                                   dtype=np.float32)
                    lmk = lmk.reshape((5, 2))
                    if 'cropped_images' in img_root:
                        file_name = os.path.join(img_root, str(subject_id), file_name)
                    else:
                        file_name = os.path.join(img_root, file_name)
                    if int(_id) == template_id and int(_media) == media_id:
                        if file_name not in id_filename_pair_img[subject_id]:
                            id_filename_pair_img[subject_id].append({"filename": file_name,"lmk": lmk})
                        Next=False
                    t_m_current_position = file_t_m.tell()
                    landmark_current_position = file_landmark.tell()

            t_m_current_position = t_m_gallery_position
            landmark_current_position = landmark_gallery_position
            read_gallery(ijbc_probe,file_t_m, file_landmark, False)

def create_dir():
    for s_id, filename_list in id_filename_pair.items():
        processed_img_subject_id = os.path.join(processed_img_root,str(s_id))
        os.makedirs(processed_img_subject_id, exist_ok=True)
        for filename in filename_list:
            shutil.copy2(filename['filename'], processed_img_subject_id)

def save_txt():
    with open('still_img.txt', 'w', encoding='utf-8') as file:
        for s_id, filename_list in id_filename_pair_img.items():
            for filename in filename_list:
                file.write(f"{s_id} {filename}\n")

    # The identity id and filename are stored in the subject id filename.txt
    with open('subject_id_filename.txt', 'w', encoding='utf-8') as file:
        for s_id, filename_list in id_filename_pair.items():
            for filename in filename_list:
                # Write key/value pairs to a file, each on its own line, separated by a space
                file.write(f"{s_id} {filename}\n")

def save_plk():
    with open(still_img_id_filename_pair_file, 'wb') as handle:
        pickle.dump(id_filename_pair_img, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(id_filename_pair_file, 'wb') as handle:
        pickle.dump(id_filename_pair, handle, protocol=pickle.HIGHEST_PROTOCOL)

os.makedirs(processed_img_root, exist_ok=True)


'''
   Run directly after modifying the path
'''

# get_img()
# create_dir()
img_root = processed_img_root
t_m_current_position = 0
landmark_current_position = 0
id_filename_pair_img = {}
id_filename_pair = {}
get_img()



