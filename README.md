
# Robbert's Note on Metadata REDCap + Metadata Tobias 

--
Multiple hierarchy of identifiers, identifying patients, visits, eyes, images uniquely 
“{Medical record Number}”  
= patients 
= +- 750 patients 
“{Medical record Number}_{Date when color photos}” 
= visit 
= +-2181 visits as ±2.94 visits x 742 patients
“{Medical record Number}_{Date when color photos}_{laterality}” 
= eye 
= ±4100 eyes (1.90-ish eyes per visit depending on the modality, ideal = 2.00 eyes per visit, if for every visit always both eyes is images)
“{Medical record Number}_{Date when color photos}_{laterality}_{filepath}”
= images 
= ± 28k images = +- 7 Topcon image per eye per visit
usefull for counts also, df.groupby(“{Medical record Number}_{Date when color photos}").count()) -> counts the visits

——
2 files:
CLINICAL METADATA: from REDCAP, contains clinical data like stage.
CSV: from REDCAP
Columns to match
Medical record number
Date 
'Date when color photos’ was obtained’
if missing fill in with 'Date when spectralis OCT was obtained’
Split right and left eye, to make every row in RETCAP an eye (and not a visit anymore)
Laterality: OD and OS
Identifier to match on : “{Medical record Number}_{Date when color photos/OCT}_{laterality(OD/OS)}” 
 Columns to keep
Event Name
Site of inclusion
Stage
Stage OD_FINAL 
Stage OS_FINAL
Stage EYE_OF_INTEREST_FINAL
IMAGE METADATA: New metadata of Tobias = image information only -> and column “filepath” which allows us to find the image -> to run our models
First get acces to Harmony ongoing -> mail Milen/Tobias last Thursday.
Mount this drive from the data server xxx/mnt/harmony_ongoing/metadata_raw_ongoing.tsv = 4.1M updated recently 
CSV Path: xxx/harmony_ongoing/metadata_raw_ongoing.tsv
Columns to Match
MRN: column=“mrn”
to match with “Medical record number” in RedCap
date: column=“HRM_ExamDate”
to match with 'Date when color photos’ was obtained’ in REDCap - if this value is missing test 'Date when spectralis OCT was obtained’
laterlality: column=“HRM_eye" = OD or OS = to match with Stage OS or OD
Identifier to match on : “{mrn}_{HRM_ExamDate}_{HRM_eye(OS/OD)}
Columns to keep
“FileName” = XXX.jpg/ YYY.png
“FileNameFullPath”
create actuall full path
concat 1) root directory with all the images + 2) value of “FileName” column
“{XX/harmony_ongoing/images}/{FileName}” 
“HRM_Manufacturer"
Merge:
“{Medical record Number}_{Date when color photos}_{laterality}_{image_filepath}” = images 
Use margin of 30 days between matching of 1) date in REDCAP csv and 2) date of fundus imaging in Tobias
select only Topcon and Zeiss images: 
“HRM_Manufacturer" == Topcon or Zeiss

Final CSV after merging
4 identifiers (patients, visit, eye, images)
“{Medical record Number}”  
“{Medical record Number}_{Date when color photos}” 
“{Medical record Number}_{Date when color photos}_{laterality}” 
“{Medical record Number}_{Date when color photos}_{laterality}_{filepath}”
“Laterality” only (just for visual checks as you can often check if the disk is on the left or right side of the image. (Right eye has disc on the right, macula on the left))
“FileNameFullPath”
“EYE_OF_INTEREST_FINAL”

From this CSV get numbers: count, patients, visits, eyes
per
total = x patients, y visits, z eyes
patients, eyes: total per event = follow-up years, x baseline, y follow-up 1...
patients, eyes: total per stage: x early, y intermediate...
patients, eyes: per event / per stage. 
rows = follow-up time
Columns = stage
2 values per cell patients/eyes
Next:
Filter on quality per image (models)
split on patient level for train-test-val)
