# Copyright © 2019 by Spectrico
# Licensed under the MIT License
import pkg_resources

model_file = pkg_resources.resource_filename('VehicleDetectionTracker', "data/model-weights-spectrico-mmr-mobilenet-128x128-344FF72B.pb")  # path to the car make and model classifier
label_file = pkg_resources.resource_filename('VehicleDetectionTracker', "data/model_labels.txt")   # path to the text file, containing list with the supported makes and models
input_layer = "input_1"
output_layer = "softmax/Softmax"
classifier_input_size = (128, 128)  # input size of the classifier