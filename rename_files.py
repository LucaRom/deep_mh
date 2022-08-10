# importing os module
# Quick rename fonction for st_johns files
import os
 
# Give the folder path 
folder = "/mnt/SN750/00_Donnees_SSD/256_over50p/twi"

#list des fichiers
#print(os.listdir(folder))

# for count, filename in enumerate(os.listdir(folder)):
#     src = f"{folder}/{filename}"
#     new_name =f"{folder}/{filename[41:-15]}" + '.las'

#     print("old name : ", src, " New name : ", new_name)
#     #os.rename(src, new_name)

# Renaming part of files
#for count, filename in enumerate(os.listdir(folder)):
#    src = f"{folder}/{filename}"
#    new_filename = filename.replace("multi_label", "mask_multiclass")
#    new_name =f"{folder}/{new_filename}"

#    #print("old name : ", src, " New name : ", new_name)
#    os.rename(src, new_name)

# Rename entire file
# for num, file in os.listdir(folder):
#     os.rename(file, "mask_bin." + num)

for num, file in enumerate(sorted(os.listdir(folder))):
    ori_file = os.path.join(folder, file)
    new_file = os.path.join(folder, "twi." + str(num) + ".tif")

    print(ori_file, new_file)

    os.rename(ori_file, new_file)