#!/bin/bash
#SBATCH --job-name=ingest_criteo
#SBATCH --partition=low        
#SBATCH --nodes=1              
#SBATCH --mem=32G              
#SBATCH --time=04:00:00        
#SBATCH --output=ingest_%j.log 

echo "Job started on $(hostname)"

module load python

python -u cluster_ingest.py