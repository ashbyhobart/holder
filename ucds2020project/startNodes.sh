#!/bin/bash
HOSTS="10.0.0.201 10.0.0.152 10.0.0.200 10.0.0.85 10.0.0.116 10.0.0.128"
SCRIPT="ray stop && ray start --address='10.0.0.151:6379' --redis-password='5241590000000000' && exit"
#SCRIPT="sudo reboot"
for s in ${HOSTS} ; do
	ssh ubuntu@${s} -o StrictHostKeyChecking=no "${SCRIPT}" 
done

