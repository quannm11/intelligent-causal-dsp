#!/bin/bash
#SBATCH --job-name=eda_criteo
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --mem=16G             
#SBATCH --time=00:30:00       
#SBATCH --output=analysis_%j.log

echo "🚀 Starting Analysis on $(hostname)"

module load python
pip install --user seaborn polars matplotlib
python eda_rawdata.py