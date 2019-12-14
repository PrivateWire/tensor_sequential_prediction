# tensor_sequential_prediction

### Overview:
Simple ML prediction allow simple number prediction from given set of inputs.

### In Order to run:

1. Clone/Download
2. Open in IDE
3. Run in browser and see results in console.
4. Change ML model parameters and refresh browser to see output.

### Changing ML model:

1. Training data inputs in xs variable. Changing this will train the new inputs these can be from a spreadsheet or or API data (ensure it is in the same format as expected by tensor). 

```javascript
const xs = tf.tensor2d([[0,0],[0.5,0.5],[1,1]]);
```
2. Outputs. This is the sample data expected as output from the neural network once trained.

```javascript
    const ys = tf.tensor2d([[1],[0.5],[0]]);
```
3. Loss alogorithm. Change this as needed - see Tensor Flow JS documentation. 
In this case  mean squared error is an estimator measure the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.

```javascript
tf.losses.meanSquaredError
```
4. Learning rate: Change value in code below. Here, we use gradient descent to update the parameters of our model which will intern adjust the weights in the neural network minimising the cost functions associated with finding the most accurate predictions and improve a the learning rate.

```javascript
    const sgdOpt = tf.train.sgd(0.5);
```
5. Epochs. This is the training iterations, so in one iteration all samples are iterated over. In tensorflows train-functions will allow you to define the value for the parameter epochs, which determines how many times your model should be trained on your sample data.  
 
