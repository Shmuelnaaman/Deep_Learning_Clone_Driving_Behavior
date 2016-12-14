# Deep_Learning_Clone_Driving_Behavior
Driving agent 

For this project, I generated a model that clone driving behavier.

model.py - The script used to create and train the model.

drive.py - The script to drive the car. You can feel free to resubmit the original drive.py or make modifications and submit your modified version.

model.json - The model architecture.

model.h5 - The model weights.

README.md - explains the structure of the network and training approach. 

For this project, I build a neural network that will learn and close driving behavior. After training, the neural network will use to drive a car in a simulator. Toward that, I choose to use the outline of the AlexNet. This is a well-known network that is not very complex and can be trained in a reasonable short time. I modify the parameter setting of the model and make it simpler. The main idea was to make sure that the training will be possible on my machine considering the large training set needed. 
After I had a working network that is complex enough but on the other hand not too complex I start gathering data. I did find that the final parameter settings of the model can vary greatly but still the network will provide a good solution. The main point was gathering good training set. 
Training, I started with 8 Epochs but soon I find out that after the 6 epoch the performance improvement of the model is not significant, that is especially true when the training set increase.


Data Exploration
Images :
The upper 3/8 part of each image was crop. That allow the model to take as an input only the road and ignore the view that is not relevant for the driving. 
Each image was down sampled by 2. That allow more complex network ( in terms of memory) with out compremising about the results so much. 

Steering Angle :
Let us examine the Steering Angle distribution using a Log scale Histogram. 

As can be seeing the zeros "Steering Angle" Frequency is larger than the other Steering Angle by at least 1 order of magnitude. Therefore we multiply the occurance of the non zeros Steering Angle by 4. 



Network Training and validation

Network Architecture

The basic structure was based on AlexNet, 3 2-D convolution layers followed by 3 perceptron network. 

The first conv layer, 

Input shape : 50 X 160 X 3

Kernel dimenssion : 48 X 5 X 11, I set the dimenssion of the model with similar proportion of the images. 

This i layer use a 'linear' activation. 

In this layer I also implement the normalization. that will make sure that the network will converage to the solution faster. 

The layer also use a MaxPooling2D of 4 X 4

The secound conv layer, 

Kernel dimenssion : 126 X 3 X 5 

This i layer use a 'linear' activation. 

The layer also use a MaxPooling2D of 3 X 3

The third conv layer, 

Kernel dimenssion : 256 X 2 X 3 

This i layer use a 'ELU' non linear activation. The idea was to set the non linearity on the largest layer. 

The layer also use a MaxPooling2D of 2 X 2

Flatening layer

The forth perceptron layer, 

Dimenssion : 512

This i layer use a 'linear' activation. 

Drop out of 0.5 to reduce overfitting 

The fifth perceptron layer, 

Dimenssion : 256

This i layer use a 'linear' activation. 

Drop out of 0.25 to reduce overfitting 

The fifth perceptron output layer, 

Dimenssion : 1

This i layer use a 'linear' activation. 

Total params: 949087

Training / Validation / Testing

For the training I use 40 K images obtained from the simulator. As the first base I use 10k images, I trained and validate the model. The Next step I use iteration to correct specific mistakes that the agent had. I was running the model, and wait for the agent to run out of the road. I then include a correction for the agent that specifically addressed the mistake that was done. I repeat that few times till the agent stays on the road for one cycle.

The model use loss function for the 'mse' since this is a regression problem and 'adam' optimizer that save us the troubles for looking after the correct learning rate.




