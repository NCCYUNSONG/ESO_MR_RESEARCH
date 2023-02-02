import os

import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
import numpy as np

def read_nii(name):
    image = sitk.ReadImage(name)
    arr = sitk.GetArrayFromImage(image)
    return arr


paramPath = 'radiomics-MRI.yaml'  # to import parameters

extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)

df = pd.DataFrame()

folder_path = "/media/liu/Stack/MR/MR1118/NII/"  # Specify the .nii folder path
label_path = "/media/liu/Stack/MR/MR1118/MASK/"

folderlist = os.listdir(folder_path)
total_num_folder = len(folderlist)
print("This check totally have %d folders" % (total_num_folder))

folderlist

new_folder_list = []
for n in folderlist:
    new_folder_list.append((n))
folderlist = sorted(new_folder_list)
folderlist = list(map(str, folderlist))

print(folderlist)  # list file names in order

df = pd.DataFrame()

for folder in folderlist:
    imager_arr = read_nii(folder_path + folder + '/trans.nii.gz')
    IName = folder_path + folder + '/trans.nii.gz'
    label_name = [str(folder) + 'GTV.nii.gz']
    for name in label_name:
        if os.path.exists(label_path + name):
            label_arr = read_nii(label_path + name)
            LName = label_path + name
            break
    if imager_arr.shape != label_arr.shape:
        print(folder)
        print(imager_arr.shape, label_arr.shape)
    print(folder)  # check if 1 MR corresponding to only 1 segmentation

    featureVector = extractor.execute(IName, LName)  # start modifiction
    df_add = pd.DataFrame.from_dict(featureVector.values()).T

    df_add.columns = featureVector.keys()
    df_add.insert(0, 'ID', str(folder))
    df = pd.concat([df, df_add])
df.to_excel('MR_feature.xlsx')