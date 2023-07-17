# NeuralNetworks

## REGRESSION
Helps us to predict a dependent output variable based on the values of independent input variable
## LINEAR REGRESSION
finds the relation between input and output variable  by plotting a line which best fits the data given to it
equation which is used to predict that is a straight line equation.

| x   |  y |
|-----|----|
| 140 | 20 |             
| 150 | 40 |
| 160 | 30 |
| 170 | 55 |

 what will be the value at 165- ?  by drawing a line of best fit through the data point we can actually predict the value at 165
 
## LOGISTIC REGRESSION
predicts value of output based on input it's output is 0 or 1 (YES OR NO).

<img src="https://github.com/manish0kuniyal/NeuralNetworks/assets/110035752/5d2acd0e-200d-4306-83d4-132085914be4" width="300"/>

 **b0 and b1 are intercept values**

## PARAMETERS

<img src="https://github.com/manish0kuniyal/NeuralNetworks/assets/110035752/3b9caa2c-67e7-4f1e-b373-a4be6779776f" width="600"/>

 ## ACTIVATION FUNCTION 
***The activation function determines whether the neuron should be "activated" (fire) or not, based on the input it receives.***

<img src="https://github.com/manish0kuniyal/NeuralNetworks/assets/110035752/faee99f8-f3a6-4515-ba9d-25b6e8cfaefb" width="600"/>


| Activation Function | Description | Use Cases |
| --- | --- | --- |
| Sigmoid | Squashes input values between 0 and 1 | - Estimating probabilities <br> - Outputting values between 0 and 1 |
| Tanh | Squashes input values between -1 and 1 | - Introducing non-linearities <br> - Normalizing data between -1 and 1 |
| ReLU | Keeps positive values as they are, turns negatives to 0 | - Hidden layers in deep neural networks <br> - Faster learning and efficient training |
| Leaky ReLU | Similar to ReLU, but allows a small negative slope | - Preventing dead neurons <br> - Improved training with negative inputs |
| Softmax | Converts values into a probability distribution | - Multi-class classification <br> - Identifying the most probable class |
