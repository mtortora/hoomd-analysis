#!/bin/bash

if [ "$#" -ge "3" ]; then
	qsub --N=$1_$2_$3 -v ALPHA=$1,P_MIN=$2,P_MAX=$3 hoomd_arcus-b.sh ${@: 4}

else
	echo -e "\033[1;31mUsage is $0 alpha p_min p_max [opt]\033[0m"
	exit

fi
