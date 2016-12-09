#FCdjz!UY

from collections import namedtuple
import csv
import glob
import os
import sys
import pycrfsuite

inputDir=sys.argv[1]
outputDir=sys.argv[2]
outPutFile=sys.argv[3]

###code to get DialogUtterance from dir
def get_utterances_from_file(dialog_csv_file):
    reader = csv.DictReader(dialog_csv_file)
    return [_dict_to_dialog_utterance(du_dict) for du_dict in reader]

def get_utterances_from_filename(dialog_csv_filename):
    with open(dialog_csv_filename, "r") as dialog_csv_file:
        return get_utterances_from_file(dialog_csv_file)

def get_data(data_dir):
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for dialog_filename in dialog_filenames:
        yield get_utterances_from_filename(dialog_filename)

DialogUtterance = namedtuple("DialogUtterance", ("act_tag", "speaker", "pos", "text"))

PosTag = namedtuple("PosTag", ("token", "pos"))

def _dict_to_dialog_utterance(du_dict):
    for k, v in du_dict.items():
        if len(v.strip()) == 0:
            du_dict[k] = None
    if du_dict["pos"]:
        du_dict["pos"] = [
            PosTag(*token_pos_pair.split("/"))
            for token_pos_pair in du_dict["pos"].split()]
    return DialogUtterance(**du_dict)

###get all labels from train or test data and create a list of it
def getLabels(utterances):
    inner_list=[]
    for utter in utterances:
        if (utter[0] != ""):
            inner_list.append(utter[0])
        else:
            inner_list.append("UNKNOWN")
    return inner_list

###get all features from train or test data and create a list of it
def getFeatures(utterances):
    b_list = []
    line_no = 1
    speaker_list = []
    for utter in utterances:
        a_list=[]
        if (line_no == 1):
            a_list.append("FU")
        speaker = utter[1]
        if (line_no == 1 or line_no == 2):
            speaker_list.append(speaker)
        else:
            speaker_list.pop(0)
            speaker_list.append(speaker)
        if (len(speaker_list) == 2 and speaker_list[0] != speaker_list[1]):
            a_list.append("spearker_changed")
        token_pos_list=utter[2]
        if (token_pos_list != None):
            for token_pos in token_pos_list:
                token = token_pos[0]
                token_string="TOKEN_" + token
                pos = token_pos[1]
                pos_string = "POS_" + pos
                a_list.append(token_string)
                a_list.append(pos_string)
            line_no += 1
        else:
            a_list.append("")
            line_no += 1
        b_list.append(a_list)
    #print(b_list)
    return b_list


trainDocFeature=list(get_data(inputDir))
trainDocLabel=list(get_data(inputDir))


##getting all feature in X and labels in Y for train data
X_train=[getFeatures(a) for a in trainDocFeature]
Y_train=[getLabels(a) for a in trainDocLabel]

##getting test data fro second directory
testDocLabel=list(get_data(outputDir))

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, Y_train):
    trainer.append(xseq, yseq);

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 150,  # stop earlier

    'feature.possible_transitions': True
})

trainer.params()
trainer.train('hw3')
tagger = pycrfsuite.Tagger()

tagger.open('hw3')

##Tagging data for test directory in X and getting all labels in Y
Y_test = [getLabels(a) for a in testDocLabel]


predictedLabels =[]
all_files=sorted(glob.glob(os.path.join(outputDir, "*.csv")))
writeOP=open(outPutFile,'w',encoding='latin1')
for file in all_files:
    writeOP.write("Filename=\""+os.path.basename(file)+"\"")
    writeOP.write("\n")
    temp = tagger.tag(getFeatures(get_utterances_from_filename(file)))
    # predictedLabels.append(temp)
    for a in temp:
        writeOP.write(a+"\n")
    writeOP.write("\n")


# correctlyMarkedLabel = 0
# totalLabelCount = 0
# for  i in range(len(predictedLabels)):
#     totalLabelCount=totalLabelCount + len( predictedLabels[i])
#     for j in range (len(predictedLabels[i])):
#         if Y_test[i][j] == predictedLabels[i][j]:
#             correctlyMarkedLabel= correctlyMarkedLabel + 1
# 
# calculateAccuracy = (correctlyMarkedLabel / totalLabelCount) * 100
# print (calculateAccuracy)



# Number of correct labels: 11021/15268
# Accuracy: 72.1836520828%
