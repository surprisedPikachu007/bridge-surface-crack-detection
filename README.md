# Surface Crack Detection
This is a deep learning model to detect surface cracks in bridges. The model is trained on the [surface crack dataset](https://www.kaggle.com/datasets/xinzone/surface-crack). This model is trained using transfer learning on the MobileNetV2 architecture.

The solution uses Transfer Learning to train the model. Transfer Learning is a technique that makes use of a pre-trained model to train a new model. Effectively, the pre-trained model acts as a general representation of the visual world. The pre-trained model is used as a feature extractor and the new model is trained on the extracted features. This technique is used to reduce the training time and to improve the accuracy of the model. We can take advantage of the already learned feature maps.

In the 'Bridge Surface Crack Detection' problem, the MobileNetV2 architecture is used as the pre-trained model. The MobileNetV2 architecture is a convolutional neural network that is trained on the ImageNet dataset.
# Steps
1. Clone the repository
2. Install the required packages
3. Run 'bridge_crack_detection.py' or 'bridge_crack_detection.ipynb'
4. A 'model.h5' file will be created in the same directory
5. Run 'judging_metrics.py' to test the model

# Usage
Example usage of the model is shown in the 'judging_metrics.py' file. The model can be used to predict the presence of cracks in a bridge image. The Precision, Recall and F1 score of the model is stored in the 'judging_metrics.txt' file.

# Results - Judging Metrics
## Model with 100 epochs of training

Tested on the test dataset

Precision: 0.9900990099009901

Recall: 1.0

F1: 0.9950248756218906

## Model with 200 epochs of training

Tested on the test dataset

Precision: 1.0

Recall: 1.0

F1: 1.0