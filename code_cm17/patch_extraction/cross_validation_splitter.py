import os 
from operator import itemgetter 
import glob
from sklearn.model_selection import KFold
import pandas as pd

save_path = '../../data/raw-data'
tumor_type = 'whole'
SPLITS =5

cm16_train_test_cm17_train_save_dir = os.path.join(save_path, 'cross_val_splits_%d_%s'%(SPLITS,tumor_type))
if not os.path.exists(cm16_train_test_cm17_train_save_dir):
    os.makedirs(cm16_train_test_cm17_train_save_dir)
    
cm16_train_test_cm17_patient_list = glob.glob(os.path.join(save_path,'**','*.svs'),recursive=True)
cm16_train_test_cm17_mask_list = glob.glob(os.path.join(save_path,'**','*%s*.tiff'%(tumor_type)),recursive=True)

print(len(cm16_train_test_cm17_patient_list),len(cm16_train_test_cm17_mask_list))

    
skf = KFold(n_splits=SPLITS)
counter = 0
for train_index, test_index in skf.split(cm16_train_test_cm17_patient_list, cm16_train_test_cm17_mask_list):
    print("TRAIN:", train_index, "TEST:", test_index)    
    image_train, image_test = itemgetter(*train_index)(cm16_train_test_cm17_patient_list), itemgetter(*test_index)(cm16_train_test_cm17_patient_list) 
    mask_train, mask_test = itemgetter(*train_index)(cm16_train_test_cm17_mask_list), itemgetter(*test_index)(cm16_train_test_cm17_mask_list) 
    
    train_data = {'Image_Path': image_train ,'Mask_Path': mask_train}
    df_train = pd.DataFrame(train_data)
    # print (df_train)
    df_train.to_csv(os.path.join(cm16_train_test_cm17_train_save_dir, 'training_fold_{0}.csv'.format(counter)), index = False)
    validation_data = {'Image_Path': image_test, 'Mask_Path': mask_test}
    df_validation = pd.DataFrame(validation_data)
    df_validation.to_csv(os.path.join(cm16_train_test_cm17_train_save_dir, 'validation_fold_{0}.csv'.format(counter)), index = False)
    # print (df_validation)
    counter += 1
