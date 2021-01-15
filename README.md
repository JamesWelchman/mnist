

# Zero third party library mnist challenge

NOTE: Currently not working - not sure why.

The design of the net is taken from [here][1].

## Running the code

```bash
	$ git clone https://github.com/JamesWelchman/mnist.git
	$ cd mnist
```

### Download the data

Hint: Copy + Paste this.
```
TRAIN_IMAGES=train-images-idx3-ubyte.gz
TRAIN_LABELS=train-labels-idx1-ubyte.gz
TEST_IMAGES=t10k-images-idx3-ubyte.gz
TEST_LABELS=t10k-labels-idx1-ubyte.gz
curl http://yann.lecun.com/exdb/mnist/$TRAIN_IMAGES > $TRAIN_IMAGES
curl http://yann.lecun.com/exdb/mnist/$TRAIN_LABELS > $TRAIN_LABELS
curl http://yann.lecun.com/exdb/mnist/$TEST_IMAGES > $TEST_IMAGES
curl http://yann.lecun.com/exdb/mnist/$TEST_LABELS > $TEST_LABELS
```

### Run the code

```bash
	$ cargo run --release
```

[1]: https://keras.io/examples/vision/mnist_convnet/