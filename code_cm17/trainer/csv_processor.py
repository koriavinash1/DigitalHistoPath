import pandas as pd
import os

# df = pd.read_csv('./train_annotation_list.csv')

# for i in range (len(df['Image_Path'])):
# 	dirname = os.path.dirname(df['Image_Path'][i])
# 	patient_name = os.path.basename(df['Image_Path'][i])
# 	patient_no = int(patient_name.split('.')[0].split('_')[1])
# 	folder = 'center_'+str(int(patient_no//20))
# 	corrected_path = os.path.join(dirname, folder, patient_name)
# 	# print (corrected_path)
# 	df['Image_Path'][i] = corrected_path
df = pd.read_csv('./annotated_train_data.csv')
for i in range (len(df['Image_Path'])):
	image_path = df['Image_Path'][i]
	label_path = df['Mask_Path'][i]
	if label_path == 'empty': 
		print (label_path)