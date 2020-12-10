#!/bin/bash
HOSTS="10.0.0.201 10.0.0.152 10.0.0.200 10.0.0.85 10.0.0.116 10.0.0.128"
SCRIPT="sudo apt-get -y update && sudo apt-get -y install python3-pip && cd DistributedKernelShap/ && pip3 install . && cd .. && exit"
#SCRIPT="ray stop && sudo pip3 uninstall -y shap && pip3 install 'shap==0.35.0' && sudo pip3 install 'shap==0.35.0' && exit"
for s in ${HOSTS} ; do
	ssh ubuntu@${s} -o StrictHostKeyChecking=no "${SCRIPT}" 
done

