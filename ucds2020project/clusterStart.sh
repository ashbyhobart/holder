#!/bin/bash
HOSTS="10.0.0.201 10.0.0.152 10.0.0.200 10.0.0.85 10.0.0.116 10.0.0.128"
#SCRIPT="sudo apt-get -y update && sudo apt-get -y install python3-pip && rm -rf holder/ && git clone https://github.com/ashbyhobart/holder.git && cd holder/ucds2020project/ && pip3 install . && exit"
SCRIPT="sudo pip3 install ray && pip3 install ray && exit"
for s in ${HOSTS} ; do
	ssh ubuntu@${s} -o StrictHostKeyChecking=no "${SCRIPT}" 
done

