# Batch vs Online Training
* It seems like batch training takes up longer time to converge. This is possibly because of the lesser number of weight updates within each epoch. Indicating that incremental training is faster.
* Online training converges in lesser number of total weight updates as compared to batch training (tested with a small batch size of 5, total number of weight updates for network to approximate function was lesser than batch updates.)
[//]: # (* SSE is not declining as smoothly as expected with batch gradient &#40;for small batch&#41; as we would expect. The SSE does not seem to flatline either.  )

[//]: # (* For full batch training, the algorithm doesn't seem to converge although the SSE flatlines after certain number of epochs. This result is not expected. With full batch, the expected gradient direction should converge toward minima. Ran the code for 10,000 epochs.)
* (Update: with full batch training, the SSE plot is always a smooth curve. However, due to pc limitations, it seems like to achieve a comparable number of weight updates, the number of epochs need to be increased drastically which is not possible with CPU.)
* Prior assumption was that batch training will take lesser number of weight updates to converge to minima
* Update: Ran batch training for 100,000 epochs which is equivalent to 1000(epochs)*100(samples) weight updates. Incremental training SSE scores drop to 0.001 whereas batch training SSE scores drop to 3.5/4.
* For reference, incremental training seems to converge less than 1000 epochs = 1000 x 100(Input samples) = 100,000 weight updates. With batch training.


# Neural Network Generaliziation
* Successfully completed generalizing stochastic gradient training to custom number of layers and custom number of transfer functions.
* Hurdles faced: Error calculation and hence sensitivity propagation was incorrect for the longest time which was causing my weights and biases to explode and activation output to overflow. Helpful note: if weights and biases seem to explode, there is a problem with sensitivity backpropagation. This should be checked first. Additionally, error calculation should be checked.
* Creating batch-training for generalized neural network was very simply since it had been done for a simple 1-S-1 network. 
* With a similar thinking, the rest of the teammates are working on variable learning, momentum learning, conjugate gradient on a simple 1-s-1 network. I will be replicating their logics into the generalized neural network training. 

# Levenberg-Marqardt Algorithm
* Extremely difficult to construct the Jakobian by following the Neural Network Design textbook. - Update: Have done this, will have to double-check on the calculation- Will use the textbook example to cross-check logic.
* Added capability of entering custom weight and bias values for debugging (compared to textbook)
* Parameter updates are also difficult as change in parameter updates (delta_x) are stored in a single array. Will have to update extract updates of each parameter and update accordingly.
* Successfully created the Jakobian!!
* Now all thats left to is to extract the parameter updates from the delta_x and follow the pseudo code.

* Successfully developed levenberg marqardt algorithm. However, for the function approximation dummy problem, the algorithm is getting stuck in a local minima or saddle point as the SSE doesn't seem to go down much once there. 
* Experimenting by tweaking, mu_max, convergence criteria, number of iterations
* Algorithm was able to converge in mere 12 epochs after different weight and bias intitialization was chosen!!
* I was using the same seed repeatedly and therefore, the weights and biases were initializing to the same values and getting stuck in the same local minima/saddle point.
* Uncertainty regarding what stopping criteria to use. As of now, SSE value < 1 is the criteria but nothing seems to converge. Stopping criteria with norm of delta_x < 0.1 also doesn't converge. Deciding these will be key steps in completing algorithm. For time being, provision has been made in making them custom set throsholds. 

# Modeling
* Mapping issues with the code.
* X_train and y_train need to strictly follow shape rules and only numerical value rules. They are not readily usable after train_test_split.
* The X features have been standardized, the y labels need to be standardized as well otherwise weights and biases are exploding.
* Going to stop mdoeling analysis here due to project deadline. Will continue to work on this 
* Update: Turns out my assumption of the dataset being cleaned was wrong. My dataset has missing values which should've been cleaned by Aditya Nayak. This is resulting in code breaking. 
* I will have to perform everything from my end once after this project submission. 

# Extra features
* I would like to add a verbose type feature in my custom training codes so that the status of training can be viewed easily.
* Create an interactive dash enable dashboard that will display the findings of this project effectively.
* LM Algo - After observing the LM algo issue, it has become clear how it is important to initialize the network with different weights and biases to get optimum results and avoid saddle point/local minima issues. Therefore, another added feature in my code could be, multiple seed initializations and picking the lowest SSE score/scores model.
* Time complexity comparison in backend execution.
* Another added argument that takes in the metric argument - Typically SSE for classification and MSE for regression.
* Perform in-depth comparison using previously defined criterias.
* Comparison with conventional models.

