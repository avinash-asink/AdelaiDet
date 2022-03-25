#After cloning the AdelaiDet repository, make the following changes for model training on custom dataset with custom parameters:


###Update the tools/train_net.py module
1. Update the train and val dataset path in main function
2. Update the number of object classes and training configuration in setup function

###Update configs/SOLOV2/R50_3x.yaml file for model training parameters

###Update configs/SOLOv2/Base-SOLOV2.yaml file to modify the model tranining initial hyperparamters


### Create pickle of config file to be used in SOLOv2 module.
###Use demo/demo.py module to test the trained model and generate and save the pickle file to be used in SOLOv2 module.

