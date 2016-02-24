package Neural

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/gonum/matrix/mat64"
)

// Network represents the neural network with its sizes, layers,
// weights, Biases and cost function.
type Network struct {
	NumLayers int
	Sizes     *mat64.Dense
	Weights   []*mat64.Dense
	Biases    []*mat64.Dense
	TestFunc  TestCompFunc
}

var sizeFloat64 = 8
var sizeInt64 = 8

// TestCompFunc is to be defined by the user, here the user gets the output from
// the Neural Network and the desired output. The user needs to return if the
// output is acceptable or not.
type TestCompFunc func(output, desiredOutput *mat64.Dense) bool

// Init initializes the weights and Biases and set the default cost function
// of the neural network.
func (n *Network) Init(size []int) {
	n.NumLayers = len(size)

	if n.NumLayers < 2 {
		log.Fatal("Need a minimum of two layers in the network")
	}

	n.Sizes = mat64.NewDense(1, n.NumLayers, nil)
	for i, val := range size {
		n.Sizes.Set(0, i, float64(val))
	}

	// Set Weights and Biases to random values.
	n.Weights = make([]*mat64.Dense, n.NumLayers-1)
	n.Biases = make([]*mat64.Dense, n.NumLayers-1)
	for i := 0; i < n.NumLayers-1; i++ {
		n.Weights[i] = mat64.NewDense(size[i], size[i+1], nil)
		for j := 0; j < size[i]; j++ {
			for k := 0; k < size[i+1]; k++ {
				n.Weights[i].Set(j, k, rand.NormFloat64())
			}
		}

		n.Biases[i] = mat64.NewDense(1, size[i+1], nil)
		for k := 0; k < size[i+1]; k++ {
			n.Biases[i].Set(0, k, rand.NormFloat64())
		}
	}

	// Set default TestCompfunc
	n.TestFunc = nil
}

// DumpWeightsBiases dumps the weights and biases of the network onto
// a given file.
func (n *Network) DumpWeightsBiases(fileName string) {
	file, err := os.OpenFile(fileName, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}
	for i := range n.Weights {
		data, err := n.Weights[i].MarshalBinary()
		if err != nil {
			log.Fatal(err)
		}
		file.Write(data)
		data, err = n.Biases[i].MarshalBinary()
		if err != nil {
			log.Fatal(err)
		}
		file.Write(data)
	}
}

// ReadWeightsBiases reads weights and biases and sets it onto the network.
// complimentary to DumpWeightsBiases.
func (n *Network) ReadWeightsBiases(fileName string) {
	file, err := os.OpenFile(fileName, os.O_RDONLY, 0644)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}
	for i := range n.Weights {
		x, y := n.Weights[i].Caps()
		data := make([]byte, x*y*sizeFloat64+2*sizeInt64)
		file.Read(data)
		n.Weights[i].Reset()
		n.Weights[i].UnmarshalBinary(data)

		x, y = n.Biases[i].Caps()
		data = make([]byte, x*y*sizeFloat64+2*sizeInt64)
		file.Read(data)
		n.Biases[i].Reset()
		n.Biases[i].UnmarshalBinary(data)
	}
}

// FeedForward propagates the input through the network and returns the output.
func (n *Network) FeedForward(input *mat64.Dense) mat64.Dense {
	var ptr *mat64.Dense

	ptr = input
	for i := 0; i < n.NumLayers-1; i++ {
		var w mat64.Dense
		w.Mul(ptr, n.Weights[i])
		w.Add(&w, n.Biases[i])
		w.Apply(sigmoidMatrix, &w)
		ptr = &w
	}
	return *ptr
}

// SGD trains the neural network using the given set of inputs and outputs.
// eta is the desired learning rate of the network.
func (n *Network) SGD(data [][]mat64.Dense, eta float64, batchSize int, test [][]mat64.Dense) {
	if n.NumLayers < 2 {
		log.Fatal("Network not set up")
	}

	iterations := len(data) / batchSize
	if len(data)%batchSize > 0 {
		iterations++
	}

	size := n.NumLayers - 1
	for i := 0; i < iterations; i++ {
		nablaB := make([]mat64.Dense, size)
		nablaW := make([]mat64.Dense, size)

		for i := 0; i < size; i++ {
			var x, y = int(n.Sizes.At(0, i)), int(n.Sizes.At(0, i+1))
			nablaW[i] = *mat64.NewDense(x, y, nil)
			nablaB[i] = *mat64.NewDense(1, y, nil)
		}

		if (len(data) - (i * batchSize)) < batchSize {
			batchSize = len(data) - (i * batchSize)
		}

		for j := 0; j < batchSize; j++ {
			tmpNablaB, tmpNablaW := n.backpropQuadCost(data[i+j], eta)
			for k := 0; k < size; k++ {
				nablaB[k].Add(&nablaB[k], &tmpNablaB[k])
				nablaW[k].Add(&nablaW[k], &tmpNablaW[k])
			}
		}

		// Change the weights and biases of the network using the
		// cost gradients obtained.
		for i := 0; i < size; i++ {
			x, y := nablaB[i].Caps()
			for j := 0; j < x; j++ {
				for k := 0; k < y; k++ {
					n.Biases[i].Set(j, k, n.Biases[i].At(j, k)-nablaB[i].At(j, k)*eta/float64(batchSize))
				}
			}
			x, y = nablaW[i].Caps()
			for j := 0; j < x; j++ {
				for k := 0; k < y; k++ {
					n.Weights[i].Set(j, k, n.Weights[i].At(j, k)-nablaW[i].At(j, k)*eta/float64(batchSize))
				}
			}
		}
	}
	if test != nil && n.TestFunc != nil {
		correct := 0
		testSize := len(test)
		for i := 0; i < testSize; i++ {
			op := n.FeedForward(&test[i][0])
			if n.TestFunc(&op, &test[i][1]) {
				correct++
			}
		}
		fmt.Printf("Success : %v/%v\n", correct, testSize)
	}
}

// This cost function uses a quadratic method to derive the error.
// error = (1/2) * (desiredOutput - output) ^ 2.
// The derivative of this with respect to the desired output is what
// we need. (i.e (output - desiredOutput)).
func quadraticCost(output, desiredOutput *mat64.Dense) mat64.Dense {
	var error mat64.Dense
	error.Sub(output, desiredOutput)
	return error
}

// Using the given input and output (data) perform back-propagation and
// return the cost gradients.
func (n *Network) backpropQuadCost(data []mat64.Dense, eta float64) (nablaB, nablaW []mat64.Dense) {
	if len(data) != 2 {
		log.Fatal("Input and output data set mismatch")
	}

	// If the no of layers in the network is 'n' then the number of
	// connections is going to be 'n-1'. (i.e the no of weight/bias matrices).
	size := n.NumLayers - 1

	// Consider the input to be the first layer of the 'activations[]'. Hence
	// always consider a '- 1' to the index of the 'activations[]' array.
	activations := make([]mat64.Dense, size+1)
	zs := make([]mat64.Dense, size)
	activations[0].Clone(&data[0])

	// Propagate the input through the layers of the network and
	// obtain the output for each layer before (zs[]) and after
	// (activations[]) applying the activation function.
	for i := 0; i < size; i++ {
		activations[i+1].Mul(&activations[i], n.Weights[i])
		activations[i+1].Add(&activations[i+1], n.Biases[i])
		zs[i].Clone(&activations[i+1])
		activations[i+1].Apply(sigmoidMatrix, &activations[i+1])
	}

	// Create matrices which are similar to the weight and bias
	// matrices, to hold cost gradients.
	nablaW = make([]mat64.Dense, size)
	nablaB = make([]mat64.Dense, size)

	for i := 0; i < size; i++ {
		nablaW[i].Clone(n.Weights[i])
		nablaB[i].Clone(n.Biases[i])
	}

	var tmp, delta mat64.Dense

	// Using the 'costFunction' obtain the gradients of the cost
	// function for the outer most layer.
	error := quadraticCost(&activations[size], &data[1])
	tmp.Apply(sigmoidPrimeMatrix, &zs[size-1])
	delta.MulElem(&error, &tmp)

	nablaB[size-1].Clone(&delta)
	nablaW[size-1].Mul(activations[size-1].T(), &delta)

	// Obtain the gradients of the cost for all other layers.
	for i := size - 2; i >= 0; i-- {
		var sp, tmp mat64.Dense

		sp.Apply(sigmoidPrimeMatrix, &zs[i])
		tmp.Mul(&delta, n.Weights[i+1].T())
		delta.Reset()
		delta.MulElem(&tmp, &sp)
		nablaB[i] = delta
		nablaW[i].Mul(activations[i].T(), &delta)
	}

	return nablaB, nablaW
}

// Wrapper function for mat64.Dense.Apply(...).
// Applies the sigmoid function to each value of the matrix.
func sigmoidMatrix(_, _ int, v float64) float64 {
	return sigmoid(v)
}

// Wrapper function for mat64.Dense.Apply(...).
// Applies the sigmoid_prime function to each value of the matrix.
func sigmoidPrimeMatrix(_, _ int, v float64) float64 {
	return sigmoid(v) * (1 - sigmoid(v))
}

// Sigmoid function maps a given value R:(-inf, +inf) to R:(0, 1).
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
