# Batch vs Online Training
* It seems like batch training takes up longer time to converge. This is possibly because of the lesser number of weight updates within each epoch. Indicating that incremental training is faster.
* However, batch training does converge in lesser number of total weight updates as compared to incremental training (tested with a small batch size of 5, total number of weight updates for network to approximate function was slightly lesser than incremental updates.)
* SSE is not declining as smoothly as expected with batch gradient (for small batch) as we would expect. The SSE does not seem to flatline either.  
* For full batch training, the algorithm doesn't seem to converge although the SSE flatlines after certain number of epochs. This result is not expected. With full batch, the expected gradient direction should converge toward minima. Ran the code for 10,000 epochs.
* (Update: with full batch training, the sse is always a smooth curve. However, due to pc limitations, it seems like to achieve a comparable number of weight updates, the number of epochs need to be increased drastically which is not possible with CPU. This was tested on GPU of google cloud and the function was approximated by increasing number of epochs to 20,000)
* For reference, incremental training seems to converge in 1000 epochs = 1000 x 100 (Input samples) = 100,000 weight updates. With batch training. 20,000 weight updates seem to approximate the function well.


# Neural Network Generaliziation
* Successfully completed generalizing stochastic gradient training to custom number of layers and custom number of transfer functions.
* Hurdles faced: Error calculation and hence sensitivity propagation was incorrect for the longest time which was causing my weights and biases to explode and activation output to overflow. Helpful note: if weights and biases seem to explode, there is a problem with sensitivity backpropagation. This should be checked first. Additionally, error calculation should be checked.
* Creating batch-training for generalized neural network was very simply since it had been done for a simple 1-S-1 network. 
* With a similar thinking, the rest of the teammates are working on variable learning, momentum learning, conjugate gradient on a simple 1-s-1 network. I will be replicating their logics into the generalized neural network training. 