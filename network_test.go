package Neural

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/gonum/matrix/mat64"
)

func TestInit(t *testing.T) {
	var a = []int{2, 3, 4}
	net := Network{}
	net.Init(a)
	if net.NumLayers != len(a) {
		t.Error("Incorrect size of Network")
	}
	x, y := net.Sizes.Dims()
	if x != 1 || y != len(a) {
		t.Error("Incorrect size of Network")
	}
	for i := 0; i < net.NumLayers-1; i++ {
		b := net.Biases[i]
		w := net.Weights[i]
		x, y := b.Dims()
		if x != 1 || y != a[i+1] {
			t.Error("Incorrect dimensions of bias matrix")
		}
		x, y = w.Dims()
		if x != a[i] || y != a[i+1] {
			t.Error("Incorrect dimensions of weight matrix")
		}
	}
}

func TestFeedforward(t *testing.T) {
	var a = []int{2, 3, 4}
	net := Network{}
	net.Init(a)
	var ip = []float64{0.5, 0.5}
	ipMatrix := mat64.NewDense(1, 2, ip)
	m := net.FeedForward(ipMatrix)
	x, y := m.Dims()
	if x != 1 || y != a[len(a)-1] {
		t.Error("Incorrect dimensions of output matrix")
	}
}

/*
 * Test the Network for a basic XOR gate.
 */
func TestSGD(t *testing.T) {
	var a = []int{2, 3, 1}
	var eta float64 = 3

	net := Network{}
	net.Init(a)
	net.TestFunc = func(output, desiredOutput *mat64.Dense) bool {
		if math.Abs(output.At(0, 0)-desiredOutput.At(0, 0)) < 0.1 {
			return true
		}
		return false
	}

	data := make([][]mat64.Dense, 10000)
	for i := 0; i < len(data); i++ {
		data[i] = make([]mat64.Dense, 2)
		rand.Seed(time.Now().UTC().UnixNano())
		x := rand.Intn(2)
		y := rand.Intn(2)
		data[i][0] = *mat64.NewDense(1, 2, []float64{float64(x), float64(y)})
		data[i][1] = *mat64.NewDense(1, 1, []float64{float64(x ^ y)})
	}

	test := make([][]mat64.Dense, 4)
	for i := 0; i < 4; i++ {
		test[i] = make([]mat64.Dense, 2)
		test[i][0] = *mat64.NewDense(1, 2, []float64{float64(i / 2), float64(i % 2)})
		test[i][1] = *mat64.NewDense(1, 1, []float64{float64((i / 2) ^ (i % 2))})
	}

	net.SGD(data, eta, 3, test)
}
