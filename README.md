# AlexNet_CIFAR_10

The objective of this project was to implement a neural network in Pytorch that
would correctly identify the ten classes that consists of vehicles and animals
from the CIFAR10 (Canadian Institute For Advanced Research) 
dataset using the AlexNet architecture.The CIFAR10 consists of 60000 32x32 
images.

The AlexNet model, a convolutional neural network (CNN) architecture created by
Alex Krizhevsky, is one of the leading architecture for any object-detection task.
For my AlexNet model, the classifier consists of 5 convolutional layers, followed by 
batch-normalization, rectified linear unit and max-pooling layers. This is followed
by 2 dropout layers, 3 linearization layers, and 2 rectified linear unit layers.

I chosen to use the Adam optimizer as it usually performs immensely well. Combining 
both the RMSProp as well as Momentum optimizers together. 

The classifier ran for 20 epochs with a learning rate of 0.5% and a batch size of 64.
It took the model ~45 minutes to fully run, start to finish. It finished with a
very lower accuracy, I decided to try the RMSProp and SGD optimizer to see if we
can achieve a higher accuracy. In the end, we achieved an accuracy of 85% from 
the SGD optimizer.

Since the classifier takes a while to compute, I wanted to optimize it to make the
classifier run faster, I did this by freezing a few of the layers, forcing the 
network to re-use some of the layers and not compute everything from scratch. 
As a result, I was able to make the classifier run much faster, but at the cost
of the accuracy. 


