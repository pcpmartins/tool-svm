# Tool for SVM classification

With this tool one can train SVMs to create classifiers using the RBF kernel and search for the best parameters (C and Gamma).
the Evaluation process can be done with the openCV SVM autoTrain method or with our own experimental method.

## Development setup

* Ubuntu 16.04 LTS
* CodeBlocks 16.01
* OpenCV 3.2.0

## Aplication features

* 1 - Train SVM classifiers
* 2 - Exponential grid-search
* 4 - Select the training/evaluation ratio for cross validation
* 5 - Use trainAuto or experimental method for optimal param. search
* 3 - Evaluation of classifiers performance.

## Input

It must be provided two files, one with the feature vector and one with the corresponding experimental binary values.
It should also be selected an interval for evaluation that determines the folds.

## Output

This application outputs different statistical measures to help evaluate how good is the performance of a model.

![figure 1](/images/tool-svm-stats2.png)
*figure 1 - Statistical measures*

## Usage

./tool-svm [featureFile] [binaryFile] [arguments]

 ### Arguments:

* -e, number of evaluation samples(default=10): -e=100, Set number of items to 100.
* -a, automatic optimization(default=false): -a=true, automatic search for optimal C and Gamma ON.
* -h, help message.

### Example

Here goes a simple example on how to use this tool with the example files provided with this project(LBTraining.csv and LBbinary.csv).

![figure 2](/images/tool-svm-example.png)
*figure 2 - Example on using the application*
