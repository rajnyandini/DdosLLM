conda activate base

pythom -m scripts.generate_carpet_bombing_data.py --dataset "A"
python -m scripts.generate_carpet_bombing_data.py --dataset "B"

python -m code.FlowSequentializer # Make DatasetA.pt file
python -m code.zero_shot_gen # Make DatasetB.pt file