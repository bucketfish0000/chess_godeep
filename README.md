The goal of this toy project is to train a neuro-network on a set of chess games to make it estimate the relative advantages/chances of winning for each player given any chessboard. Data of chess games in pgn format as well as methods to process them into boards are from <https://github.com/osipychev/chess_godeep>.

Logs:

### 15 March

Pre-processing is handled by `reader.py`. Run the following to get data ready for training:

    importlib.reload(reader.py)
    white,black,sets = reader.readtxt()

A board is divided into 6 channels, each contains info about positions of a single kind of chess piece (e.g. all pawns are in one channel, all bishops another).

Original thoughts were also to keep additional channels for the original board as well as combinations of piece categories(e.g. minor pieces are pawns and bishops together), but this seems to be redundant, thus a cut-off.

Apparantly chars cannot be fed into `torch.nn` Neuro Nets, so numbers are used instead. Black pieces are represented by negative numbers, White pieces positive.

Results of games are (for now) represented in two-tuples: (1,0) means White winning, (0,1) Black winning. The idea is to associate each board with the result of the game, in the process of which the board appeared.

Ambiguity shows up here:

- In this manner of training, the outcome of the Neuro Net would be also a two-tuple (each number in which is likely to be something near the range(0,1), although sometimes negative). The meaning of it is not clear, which would make it hard to define an appropriate loss criteria.

- Boards that typically occur in the opening of a game tend to be associated to both kind of results. However, notice that the advantage/chance of winning on such boards are ambiguous is ambiguous itself, this seems to be tolerable.

Other problems include:

- The giant size of data, which makes the pre-processing very long(typicallt ~10 mins). The way of handling probably contributed to this; there should be faster ways to do it.

Each piece of data in the form of (board, tuple) is randomly put into one out of ten sets, for the purpose of 10-Fold Cross Validation.

---

### 16 March

Did nothing.

---

### 17 March

Training is handeled by `training.py`. Run the following to train the NN over the data:

    importlib.reload(reader.py)
    model, loss_fn, optimizer = training.train_loop(sets,epoch=<int>)

`training_loop` goes over all data for `epoch` times (default is 10).

As mentioned above, the data are divided into 10 roughly even partitions. In each iteration, 9 of them (upon random selection) are used to train the model, while the 1 leftover is used to test the performance of the trained model.

The Neuro Net configs are like this:

    (Input: 6-channel 8*8)
    torch.nn.Conv2d(6,4,2)
    (Output: 4-channel 7*7)
    ------
    torch.nn.ReLU()
    torch.nn.Conv2d(4,1,4)
    (Output: 1-channel 4*4)
    ------
    torch.nn.ReLU()
    torch.nn.AvgPool2d(2)
    (Output: 1-channel 2*2)
    (Flatten the output)
    ------
    torch.nn.Linear(4,2)
    (Output: 1-channel 1*2)

Which is rather simplistic.
The idea is to use smaller convolution-kernels to recognize small/regional features of the board, followed by larger convolution-kernels to tell patterns that are more step-back.

For loss function and optimization method, `torch.nn.CrossEntropyLoss` and `torch.optim.Adam` are used. Not using SGD as Adam seems faster.

All hyperparams, such as learning rate and epoch number, as well as the choices of non-linear functions, losses and optimizers, are selected rather casually and randomly for now.

Problems at this point:

- The output of the model does not seem to be fitting to the data very well, even at relatively large learning rate. Could be because of inappropriate choice of functions, numbers of input/output, numbers of layers and many other things.

- Still ambiguous criteria of model performance: cross entropy does not seem to be a good estimate. The form of output as a tuple is ambiguous to begin with.

- No way to tell issues such as overfitting.

- Long running time. Could be better if running on GPU (for some reason not supported on the machine working).

- Need to output and store pre-processed data as well as training results to files instead of writing to cmd lines.

------

### 18 March

Updated logs.

------
