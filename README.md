# **CNN - Convolution Neural Network**
## *Transfer learning / fine-tuning*

## **Description**

Transfer learning refers to the process of levarging the knowledge learned in one model for the training of another model, in order to save time, get mor accuracy and better response with a smaller quantity of images

### Objective
- based on the model, train a CNN using cat and dogs for classification
- Use transfer learning to improve de result

### Training information

This training process was based on the tutorial project of MNIST Dataset
https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb 

The dataset for the training is:
https://www.microsoft.com/en-us/download/details.aspx?id=54765

### Considerations

The network training process started in GOOGLE COLAB, but as I was having problems running it, including interruptions in the middle of the process, I prepared VSCODE to run the task.

All the results were obtained running in a windows 11 environment, on a local machine, making the operating system adjustments, since COLAB runs in a linux environment.


### Hardware
Intel Core i5-8265, CPU 1.60GHz
RAM: 16GB
Windows 11, x64

## Results

### Scratch training

Original qty of pictures (cats and dogs): 12.500
Selected for the training process: 20% (train and test)
one image with problem, that was rejected.

### Summary of what we have
- finished loading 4999 images from 2 categories
- train / validation / test split: 3499, 750, 750
- training data shape:  (3499, 224, 224, 3)
- training labels shape:  (3499, 2)

### Model
Sequential
- Total params: 1,209,058 (4.61 MB)
- Trainable params: 1,209,058 (4.61 MB)
- Non-trainable params: 0 (0.00 B)

Kept the same configuration proposed in the tutorial

Time to run: 12m28.9s

 Total params: 1,209,058 (4.61 MB)
 Trainable params: 1,209,058 (4.61 MB)
 Non-trainable params: 0 (0.00 B)


### Results

Test loss: 0.5364869832992554
Test accuracy: 0.7226666808128357


### Transfer training (proposed)

Model: "functional_20"

- Total params: 134,268,738 (512.19 MB)
- Trainable params: 8,194 (32.01 KB)
- Non-trainable params: 134,260,544 (512.16 MB)

Execution time: 169m22.2s

Test loss: 0.23198756575584412
Test accuracy: 0.9026666879653931

