# Batch vs Online Training
* It seems like batch training takes up longer time to converge. This is possibly because of the lesser number of weight updates within each epoch. 
* However, batch training does converge in lesser number of total weight updates as compared to incremental training. 
* SSE is not declining as smoothly as expected with batch gradient (for small batch) as we would expect. The SSE does not seem to flatline either. Ran the code for 10,000 epochs. 
* For full batch training, the algorithm doesn't seem to converge although the SSE flatlines after certain number of epochs. This result is not expected. With full batch, the expected gradient direction should converge toward minima. 

# Neural Network Generaliziation
* Successfully completed generalizing stochastic gradient training to custom number of layers and custom number of transfer functions.
* Hurdles faced: Error calculation and hence sensitivity propagation was incorrect for the longest time which was causing my weights and biases to explode and activation output to overflow. Helpful note: if weights and biases seem to overflow, there is a problem with sensitive backpropagation. This should be checked first. Additionally, error calculation should be checked.