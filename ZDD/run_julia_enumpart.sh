#!/bin/zsh

for n in 9 10 
do
	for k in $n 
	do
		for d in 0 
		do
  			sbatch kick_off_run.sh $n $n $k $d "rook" "weighted"
#			sbatch kick_off_run.sh $n $n $k $d "queen" "weighted"
 			sbatch kick_off_run.sh $n $n $k $d "rook" "weightless"
#			sbatch kick_off_run.sh $n $n $k $d "queen" "weightless"
		done
	done
done
