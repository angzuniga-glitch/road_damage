1. IDENTIFYING INFORMATION
   
a. Full Name:Tad Kitaguchi, Angel Zuniga, Khoa Nguyen
b. Student ID: 2403661, 2504359, 2457790
c. Chapman Email: tkitaguchi@chapman.edu,angezuniga@chapman.edu, khoanguyen@chapman.edu
d. Course Number and Section: CPSC542-01
e. Assignment or Exercise Number: Assignment 2

2. DATASET: RDD2022 – Road Damage Dataset
   
https://github.com/sekilab/RoadDamageDetecto

The RDD2022 dataset contains real-world road images collected from multiple countries, with annotated bounding boxes for different types of road damage. For this assignment, cropped regions of damage are used for classification.

Classes used in this project:
	D00 – Longitudinal cracks (along the road)
	D10 – Lateral cracks (across the road)
	D20 – Alligator cracks (network cracking)
	D40 – Potholes / surface damage

3. GITHUB REPO
   
https://github.com/angzuniga-glitch/road_damage.git

4. TASKS
   
a. Data Processing and Pipeline Development 
b. Model Development
c. Model Training and Evaluation
d. Analysis

5. INSTRUCTIONS FOR ACCESSING THE ASSIGNMENT
   
a. git clone https://github.com/angzuniga-glitch/road_damage.git
b. pip install -r requirements.txt
c. python -m src.train --config configs/<target_yaml_file_to_train.yaml>
d. python -m src.eval \
  --config configs/customcnn.yaml \
  --checkpoint outputs/customcnn/checkpoints/best.pt \
  --split val
