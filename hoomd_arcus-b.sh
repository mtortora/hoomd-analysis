#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --partition=compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maxime.tortora@icloud.com

module purge
module load hoomd

ALPHA=`echo ${ALPHA} | awk '{printf("%.1f\n", $1)}'`

# Submit array of 16 jobs
for ID in `seq 0 15`; do
	P_SIM=`echo ${P_MIN} ${P_MAX} $ID | awk '{printf("%.3f\n", $1+($2-$1)/16*$3)}'`

	mkdir -p a_$ALPHA/p_$P_SIM
	python hoomd_sc.py $ALPHA $P_SIM $@ > a_$ALPHA/p_$P_SIM/output.log&
done

wait
