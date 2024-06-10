# Risk Assessment of Test Set Annotation Errors for Evaluating Object Detectors
## Environment
Ubuntu 20.04.3 LTS  
Python 3.8.10  
Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
## Setup
Step 1: Get repository  
```
git clone https://github.com/SMNUResearch/ReliableObjectDetection.git
cd ReliableObjectDetection
```
Step 2: Install dependencies  
```
sh install.sh
```
## Run Simulation
Modeling corner errors
```
python3 main_corner.py -OPTION=scale
python3 main_corner.py -OPTION=shift
python3 main_corner.py -OPTION=IoU
```
Simulate corner and localization errors
```
python3 main_metric.py -OPTION=localization
```
Simulate existence errors
```
python3 main_metric.py -OPTION=existence
```
Visualize simulation results
```
python3 visualization.py -OPTION=localization
python3 visualization.py -OPTION=existence
```
