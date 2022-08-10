import numpy as np
import os
import shutil
import tifffile as tiff

def move_bad_files(in_path, out_path, bad_img_list):
    for i in bad_img_list:
        file_path = os.path.join(in_path, i)
        exist = os.path.exists(out_path)
        if exist == True:
            shutil.move(file_path, out_path)
            print(f"{i} moved")
        else:
            print("Output path doesn't exist")
            break

def check_tiles_shape(in_path, wanted_shape):
    # Shape filter on images
    # Return list of images with shape different from wanted_shape

    all_img = [x for x in os.listdir(in_path) if x.endswith(('.tif'))]

    bad_img_list = []

    for i in all_img :
        img_path = os.path.join(in_path, i)
        test_img = np.array(tiff.imread(img_path), dtype=np.float32)
        
        if test_img.shape != wanted_shape:
            bad_img_list.append(i)
            print(test_img.shape)
        else:
            pass

    return bad_img_list

if __name__ == "__main__":

    # in_path = "D:/00_Donnees/02_maitrise/01_trainings/estrie/512/sen2_ete"
    # out_path = "D:/00_Donnees/02_maitrise/01_trainings/estrie/512/z_not_512_sen"
    in_path = "/mnt/SN750/00_Donnees_SSD/256_over50p/sen2_ete"
    out_path = "/mnt/SN750/00_Donnees_SSD/256_over50p/z_not_256_sen_ete"
    #wanted_shape = (512, 512, 12)
    wanted_shape = (256, 256, 12)

    bad_img_list = check_tiles_shape(in_path, wanted_shape)
    move_bad_files(in_path, out_path, bad_img_list)

    #print(bad_img_list)
    #print(len(bad_img_list))