#!/usr/bin/env bash
echo usage: 
echo scriptName.sh : run in normal mode
echo scriptName.sh debug : run in debug mode

# hardware
cudaID=0

# debug mode
if [[ $# != 0 ]] && [[ $1 == "debug" ]]
then
    debug=true
else
    debug=false
fi

seed=4321
seed=1

# dataset
# dataDir='oos,hwu64,stackoverflow'
# dataDir="oos,hwu64,mcid"
dataDir='oos,hwu64,bank77'

sourceDomain="utility,auto_commute,kitchen_dining,work,home,meta,small_talk,travel"
# valDomain="alarm,audio,iot,calendar,play,datetime,takeaway,news,music,weather,qa,social,recommendation,cooking,email,transport,lists"
valDomain="iot,play,qa"

# below is only for evaluation
targetDomain="BANKING"
# targetDomain="STACKOVERFLOW"
# targetDomain="MEDICAL"
# targetDomain="curekart,powerplay11,sofmattress"

# setting
shot=2

# training

saveModel=--saveModel
saveName=none
validation=--validation
mlm=
learningRate=1e-5
# learningRate=5e-6


# 学习率是5e-6,2023年9月4日15:32:55感觉这个学习率有点小,之后尝试调大一下
# model setting
# common
LMName=bert-base-uncased

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    epochs=1
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
logFolder=./log/
mkdir -p ${logFolder}
logFile=${logFolder}/transfer_${sourceDomain}_to_${targetDomain}_${way}way_${shot}shot.log
# logFile=${logFolder}/transfer_mlm_${sourceDomainName}_to_${targetDomainName}_${way}way_${shot}shot.log
if $debug
then
	logFlie=${logFolder}/logDebug.log
fi

export CUDA_VISIBLE_DEVICES=${cudaID}
python IntentBert-main/transfer.py \
	--seed ${seed} \
	--valDomain  ${valDomain}  \
    	--sourceDomain ${sourceDomain} \
    	--targetDomain ${targetDomain} \
    	${tensorboard} \
    	--dataDir ${dataDir} \
    	--shot ${shot}  \
    	${saveModel} \
    	${validation} \
    	${mlm} \
        --learningRate  ${learningRate} \
    	--LMName ${LMName} \
    	--saveName ${saveName} \
    	| tee "${logFile}"
echo "Experiment finished."
