#!/bin/sh
#SBATCH -J zdd 
#SBATCH -p preempt 
#SBATCH --time=4-0:00:00
#SBATCH -n 4
#SBATCH --mem=200G
julia julia_enumpart.jl $1 $2 $3 $4 $5 $6 >> zdd_stdout.txt 
