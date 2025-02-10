import os
import pickle
import random
import cv2
from skimage import transform as trans
import numpy as np
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset,Subset
from collections import defaultdict
# to avoid ValueError: Decompressed Data Too Large
ImageFile.LOAD_TRUNCATED_IMAGES = True


def select_p_images_per_class(dataset, k):
    # Get the sample index for each class
    class_to_indices = defaultdict(list)
    for idx, target in enumerate(dataset.targets):
        class_to_indices[target].append(idx)

    # k images are randomly selected from each class
    selected_indices = []
    for indices in class_to_indices.values():
        selected_indices.extend(random.sample(indices, k))

    # Create a new subset containing the selected samples
    subset = Subset(dataset, selected_indices)

    return subset

def save_plk(gallery_file, known_file, unknown_file, probe_file, Gallery, Known, Unknown, Probe, val, test, known_id_set_len):
    with open(gallery_file, 'wb') as handle:
        pickle.dump(Gallery, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Gallery.pkl has been saved!')

    with open(known_file, 'wb') as handle:
        pickle.dump(Known, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Known.pkl has been saved!')

    with open(unknown_file, 'wb') as handle:
        pickle.dump(Unknown, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Unknown.pkl has been saved!')

    with open(probe_file, 'wb') as handle:
        # Store Probe, val, test, and known_id_set_len together in the Probe file
        pickle.dump((Probe, val, test, known_id_set_len), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Probe.pkl has been saved with val, test, and known_id_set_len!')

def load_plk(gallery_file, known_file, unknown_file, probe_file):
    with open(gallery_file, 'rb') as handle:
        Gallery = pickle.load(handle)
        print('Gallery.pkl has been loaded!')

    with open(known_file, 'rb') as handle:
        Known = pickle.load(handle)
        print('Known.pkl has been loaded!')

    with open(unknown_file, 'rb') as handle:
        Unknown = pickle.load(handle)
        print('Unknown.pkl has been loaded!')

    with open(probe_file, 'rb') as handle:
        Probe, val, test, known_id_set_len = pickle.load(handle)
        print('Probe.pkl has been loaded with val, test, and known_id_set_len!')

    return Gallery, Known, Unknown, Probe, val, test, known_id_set_len

# IJBC
t_m_current_position, landmark_current_position = 0, 0
id_filename_pair, id_filename_pair_img = {}, {}
def read_gallery(filename, file_t_m, file_landmark, img , processed_img_root):
    global t_m_current_position, landmark_current_position
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
        file_name = os.path.join(processed_img_root, str(subject_id), file_name)
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

def get_img(ijbc_t_m, ijbc_5pts, ijbc_gallery_1, ijbc_gallery_2, ijbc_probe, processed_img_root, num_gallery):
    global t_m_current_position
    global landmark_current_position
    with open(ijbc_t_m, 'r') as file_t_m:
        with open(ijbc_5pts, 'r') as file_landmark:
            read_gallery(ijbc_gallery_1, file_t_m, file_landmark, True,  processed_img_root)
            read_gallery(ijbc_gallery_2, file_t_m, file_landmark, True, processed_img_root)

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
                    # Reads from the beginning of the current position until the next newline, i.e., "current line."
                    current_line_t_m = file_t_m.readline()
                    current_line_landmark = file_landmark.readline()
                    file_name, _id, _media = current_line_t_m.strip().split(' ')
                    lmk = current_line_landmark.strip().split(' ')[1:-1]
                    lmk = np.array([float(x) for x in lmk],
                                   dtype=np.float32)
                    lmk = lmk.reshape((5, 2))

                    file_name = os.path.join(processed_img_root,str(subject_id),file_name)
                    if int(_id) == template_id and int(_media) == media_id:
                        if file_name not in id_filename_pair_img[subject_id]:
                            id_filename_pair_img[subject_id].append({"filename": file_name,"lmk": lmk})
                        Next=False
                    # Check the current pointer position (after the newline character of the line read)
                    t_m_current_position = file_t_m.tell()
                    landmark_current_position = file_landmark.tell()

            t_m_current_position = t_m_gallery_position
            landmark_current_position = landmark_gallery_position
            read_gallery(ijbc_probe,file_t_m, file_landmark, False, processed_img_root)
    return id_filename_pair_img, id_filename_pair


def partition_dataset(ijbc_t_m, ijbc_5pts, ijbc_gallery_1, ijbc_gallery_2, ijbc_probe, processed_img_root,
                      plk_file_root, num_gallery):
    print('Partition dataset...')

    # Define the path to the plk file
    gallery_file, known_file, unknown_file, probe_file = os.path.join(plk_file_root, "Gallery.plk"), os.path.join(
        plk_file_root, "Known.plk"), os.path.join(plk_file_root, "Unknown.plk"), os.path.join(plk_file_root,
                                                                                              "Probe.plk")
    if os.path.exists(gallery_file) and os.path.exists(probe_file):
        return load_plk(gallery_file, known_file, unknown_file, probe_file)

    # Gets a mapping of image filenames and ids
    id_filename_pair_img, id_filename_pair = get_img(ijbc_t_m, ijbc_5pts, ijbc_gallery_1, ijbc_gallery_2, ijbc_probe,
                                                     processed_img_root, num_gallery)

    Gallery, Known, Unknown = [], [], []
    val_K, test_K, val_U, test_U = [], [], [], []
    ten_id_set = set()
    known_id = 0

    # Iterate over id filename pair img and collect the eligible identities
    for s_id, filename_list in tqdm(id_filename_pair_img.items()):
        if len(filename_list) > 10:  # An identity with more than 10 files is satisfied
            ten_id_set.add(s_id)

    # Randomly pick half of the identities
    ten_id_list = list(ten_id_set)
    random.shuffle(ten_id_list)
    known_id_list = ten_id_list[:1765]
    known_id_set = set(known_id_list)

    for s_id, filename_list in tqdm(id_filename_pair_img.items()):
        if s_id in known_id_set and len(filename_list) > 10:
            gallery_image_list = np.random.permutation(filename_list).tolist()
            s_id_gallery = gallery_image_list[:num_gallery]
            Gallery += [(gallery['filename'], known_id, gallery['lmk']) for gallery in s_id_gallery]
            known_id += 1

    num_known = known_id

    # Construct Known and Unknown sets
    known_id = 0
    unknown_pairs_by_id = []  # Each unknown identity and its pairs data are temporarily stored
    for s_id, pairs in tqdm(id_filename_pair.items()):
        if s_id in known_id_set:
            gallery_filename_list = [g[0] for g in Gallery[known_id * num_gallery:known_id * num_gallery + num_gallery]]
            Known += [(pair['filename'], known_id, pair['lmk']) for pair in pairs if
                      pair['filename'] not in gallery_filename_list]
            known_id += 1
            random.shuffle(Known)
            half_length = len(Known) // 2

            test_K = Known[:half_length]  # 前一半数据
            val_K = Known[half_length:]  # 后一半数据

        else:
            Unknown += [(pair['filename'], num_known, pair['lmk']) for pair in pairs]
            unknown_pairs_by_id.append((s_id, pairs))

    Probe = Known + Unknown

    # Step 1: Randomly shuffle the order of all unknown identities
    random.shuffle(unknown_pairs_by_id)

    # Step 2: Calculate half of the number of unknown identity categories
    half_length_unknown = len(unknown_pairs_by_id) // 2

    # Step 3: The data of half of the identities are added to test U and the other half to val U
    for s_id, pairs in unknown_pairs_by_id[:half_length_unknown]:
        test_U += [(pair['filename'], num_known, pair['lmk']) for pair in pairs]

    for s_id, pairs in unknown_pairs_by_id[half_length_unknown:]:
        val_U += [(pair['filename'], num_known, pair['lmk']) for pair in pairs]

    val = val_K + val_U
    test = test_K + test_U

    save_plk(gallery_file, known_file, unknown_file, probe_file, Gallery, Known, Unknown, Probe, val, test,
             len(known_id_set))

    return Gallery, Known, Unknown, Probe, val, test, len(known_id_set)

class open_set_folds():
    def __init__(self, image_directory, known_list_path, unknown_list_path, num_gallery):
        with open(known_list_path, 'rb') as fp:
            known_list = pickle.load(fp)
        with open(unknown_list_path, 'rb') as fp:
            unknown_list = pickle.load(fp)
        num_known = len(known_list)

        # Gallery, Known Probe, Unknown Probe set
        self.G, self.val_K, self.val_U, self.test_K, self.test_U = [], [], [], [], []
        known_id = 0  # assign id to known identities

        for name in known_list:
            image_list = os.listdir(os.path.join(image_directory, name))
            if len(image_list) <= 10:  # cannot be used as known identity
                num_known -= 1
            else:
                image_list = np.random.permutation(image_list).tolist()  # randomly shuffle
                for i in range(num_gallery):
                    image_path = os.path.join(image_directory, name, image_list[i])
                    self.G.append((image_path, known_id))

                val_known_size = int((len(image_list) - num_gallery) / 2)

                for i in range(num_gallery, num_gallery + val_known_size):
                    image_path = os.path.join(image_directory, name, image_list[i])
                    (self.
                     val_K.append((image_path, known_id)))
                for i in range(num_gallery + val_known_size, len(image_list)):
                    image_path = os.path.join(image_directory, name, image_list[i])
                    self.test_K.append((image_path, known_id))
                known_id += 1


        val_unknown = np.array(unknown_list)[:int(len(unknown_list) * 0.5)]
        test_unknown = np.array(unknown_list)[int(len(unknown_list) * 0.5):]

        for name in val_unknown:
            image_list = os.listdir(os.path.join(image_directory, name))
            image_list = np.random.permutation(image_list).tolist()  # randomly shuffle
            for i in range(len(image_list)):
                image_path = os.path.join(image_directory, name, image_list[i])
                self.val_U.append((image_path, num_known))
        for name in test_unknown:
            image_list = os.listdir(os.path.join(image_directory, name))
            image_list = np.random.permutation(image_list).tolist()  # randomly shuffle
            for i in range(len(image_list)):
                image_path = os.path.join(image_directory, name, image_list[i])
                self.test_U.append((image_path, num_known))

        self.val = self.val_K + self.val_U
        self.test = self.test_K + self.test_U
        self.num_known = num_known  # Number of known identities


class face_dataset(Dataset):
    def __init__(self, data_fold, transform, img_size):
        self.data_fold = data_fold
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data_fold)

    def __getitem__(self, index):
        image_path, label = self.data_fold[index]
        image = Image.open(image_path).resize((self.img_size, self.img_size))
        if image.mode == 'L':
            image = image.convert("RGB")
        image = self.transform(image)
        return image, label

class ijbc_dataset(Dataset):
    def __init__(self, data_fold, transform, img_size):
        self.data_fold = data_fold
        self.transform = transform
        self.img_size = img_size

        self.aligner = ImageAligner()
        self.image_is_saved_with_swapped_B_and_R = False

    def __len__(self):
        return len(self.data_fold)

    def __getitem__(self, index):
        image_path, label, lmk = self.data_fold[index]

        img = cv2.imread(image_path)
        img = img[:, :, :3]
        img = self.aligner.align(img, lmk)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label

class ImageAligner:
    def __init__(self, image_size=(112, 112)):

        self.image_size = image_size
        src = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32)
        if self.image_size[0] == 112:
            src[:, 0] += 8.0
        self.src = src

    def align(self, img, landmark):
        # align image with pre calculated landmark

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark

        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        return img




