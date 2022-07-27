# importing os module
# Quick rename fonction for st_johns files
import os
 
# Give the folder path 
folder = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/512/mask_multiclass"

#list des fichiers
print(os.listdir(folder))

# for count, filename in enumerate(os.listdir(folder)):
#     src = f"{folder}/{filename}"
#     new_name =f"{folder}/{filename[41:-15]}" + '.las'

#     print("old name : ", src, " New name : ", new_name)
#     #os.rename(src, new_name)

# Renaming part of files
for count, filename in enumerate(os.listdir(folder)):
    src = f"{folder}/{filename}"
    new_filename = filename.replace("multi_label", "mask_multiclass")
    new_name =f"{folder}/{new_filename}"

    #print("old name : ", src, " New name : ", new_name)
    os.rename(src, new_name)
