# HandwrittenDigitClassifier
This is a feedforward neural network that identifies handwritten digits written from scratch using only math. More info in the Word doc.

Xavier Init Training Acc: ~90%
Xavier Init Dev Acc: ~90%
Record Training Acc: 100% within 20 iterations (16.9s)
Record Dev Acc: 98.2% within 100 iterations (113.0s)

Data: 42,000 28x28 Images
Architecture: 784 input nerons, 64 hidden neurons, 10 output neurons
Activation Func: ReLU, SoftMax
LR: 0.05 with Cosine Annealing down to 0.01
Lambda Reg: 0.01