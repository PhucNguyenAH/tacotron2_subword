# Inpfile="data/train_val.txt" #txt form wave_path|script
Inpfile="data/vi_dataset/script/train.txt" #txt form wave_path|script
Outdir="data/vi_dataset"
Normfile=$Outdir/norm/train.txt
Meldir=$Outdir/mels
Phonedir=$Outdir/phones

# python preprocess/getNorm.py $Inpfile $Normfile

python preprocess/getPhone.py $Normfile $Phonedir

# python get/Mel.py $Normfile $Meldir