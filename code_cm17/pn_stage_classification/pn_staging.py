import pandas as pd

# DFCN
# raw_prediction_path_EN = '../predictions/Report_20190802_113031/EnsembleOnFinalTestSet_predictions_20190802_113120.csv'
# raw_prediction_path_RF = '../predictions/Report_20190802_113031/RFOnFinalTestSet_predictions_20190802_113120.csv'

# NCRF
# raw_prediction_path_EN = '../predictions/Report_20190803_161448/EnsembleOnFinalTestSet_predictions_20190803_161649.csv'
raw_prediction_path_RF = '../predictions/Report_20190803_161448/RFOnFinalTestSet_predictions_20190803_161649.csv'

# Stage Mapping
NEGATIVE = 'negative'
ITC = 'itc'
MICRO = 'micro'
MACRO = 'macro'

def get_patient_stage(node_stages):
    stage_count = {NEGATIVE: 0, ITC:0, MICRO:0, MACRO:0}
    for stage in node_stages:
        stage_count[stage] +=1
    if stage_count[NEGATIVE] == 5:
        pn_stage = 'pN0'
    elif stage_count[MICRO] == 0 and stage_count[MACRO] == 0:
        pn_stage = 'pN0(i+)'
    elif stage_count[MICRO] >= 1 and stage_count[MICRO] <= 3 and stage_count[MACRO] != 0:
        pn_stage = 'pN1'
    else:
        pn_stage = 'pN2'

    return pn_stage

if __name__ == '__main__':
    # Submission_file_name = 'Submit_RF_DFCN_UNET'
    # Submission_file_name = 'Submit_EN_DFCN_UNET'

    Submission_file_name = 'Submit_RF_NCRF'
    # Submission_file_name = 'Submit_EN_NCRF'

    f = open('../predictions/Report_20190802_113031/{}.csv'.format(Submission_file_name), 'a')
    f.write("patient" + "," + "stage" + "\n")
    df = pd.read_csv(raw_prediction_path_RF)
    number_nodes = 5
    node_stages = []
    for index, row in df.iterrows():
        print (index, row['patient'], row['stage'])
        node_stages.append(row['stage']) 
        if (index+1) % number_nodes == 0:
            patient_name = row['patient'][0:11]
            # print (patient_name, node_stages)
            pn_stage = get_patient_stage(node_stages)
            f.write(patient_name + ".zip" + "," + pn_stage + "\n")
            for i in range(number_nodes):
                f.write(patient_name + "_node_"+str(i)+ "," + node_stages[i] + "\n")
            node_stages = []
    f.close()









# import numpy as np

# preds = np.loadtxt("predictions/run5raw.csv", dtype="int", delimiter=",")

# node_dc = {}
# pat_dc = {}


# submissionnum = 2

# f = open('predictions/submission'+str(submissionnum)+'.csv', 'a')
# f.write("patient, stage\n")

# for row in preds:
#     patient = row[0]
#     node = row[1]
#     pred = row[2]
#     ky = str(patient) + "_" + str(node)
#     if node_dc.has_key(ky):
#         node_dc[ky] += [pred]
#     else:
#         node_dc[ky] = [pred]

# node_dc["100_4"] = 0
# node_dc["140_4"] = 0
# node_dc["183_0"] = 0

# print node_dc

# for key in node_dc:
#     vals = node_dc[key]
#     mx = np.max(vals)
#     pat_ky, nod_ky = key.split("_")
#     # f = open('predictions/submission'+str(submissionnum)+'.csv', 'a')
#     # f.write("patient_" + pat_ky + "_node_" + nod_ky + ".tif," + labels[mx] + "\n")
            
#     if pat_dc.has_key(pat_ky):
#         pat_dc[pat_ky] += [mx]
#     else:
#         pat_dc[pat_ky] = [mx]

# print pat_dc

# for key in pat_dc.keys():
#     vals = pat_dc[key]
#     if len(vals) < 5:
#         print(key)
#     mx = np.max(vals)
#     count3 = 0
#     for v in vals:
#         if v == 3:
#             count3 += 1

#     if mx == 0:
#         label = "pN0"
#     elif mx == 1:
#         label = "pN0(i+)"
#     elif mx == 2:
#         label = "pN1mi"
#     elif count3 < 3:
#         label = "pN1"
#     else:
#         label = "pN2"
#     # f = open('predictions/submission'+str(submissionnum)+'.csv', 'a')
#     # f.write("patient_" + key + ".zip," + label + "\n")