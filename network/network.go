package network

import (
	"math"

	"github.com/haashi/go-neural/matrix"
)

// Network : a neural network
type Network struct {
	inputs       int
	hiddenLayers int
	outputs      int
	weights      []*matrix.Matrix
	learningRate float64
	biasWeights  []*matrix.Matrix
	bias         float64
}

// CreateNetwork : create a neural network
func CreateNetwork(inputs int, hiddenLayers int, layersSize []int, outputs int, learningRate float64, bias float64) *Network {
	seed = 1103527590
	net := Network{
		inputs:       inputs,
		hiddenLayers: hiddenLayers,
		outputs:      outputs,
		learningRate: learningRate,
		bias:         bias,
	}
	layersSize = append([]int{inputs}, layersSize...)
	layersSize = append(layersSize, outputs)
	net.weights = make([]*matrix.Matrix, hiddenLayers+1)
	net.biasWeights = make([]*matrix.Matrix, hiddenLayers+1)
	for i := 0; i < hiddenLayers+1; i++ {
		net.weights[i] = &matrix.Matrix{
			Col:    layersSize[i+1],
			Row:    layersSize[i],
			Values: initArray(layersSize[i] * layersSize[i+1]),
		}
		net.biasWeights[i] = &matrix.Matrix{
			Col:    layersSize[i+1],
			Row:    1,
			Values: initArray(layersSize[i+1]),
		}
	}

	for k := 0; k < hiddenLayers+1; k++ {
		for j := 0; j < net.weights[k].Col; j++ {
			for i := 0; i < net.weights[k].Row; i++ {
				net.weights[k].Values[i*net.weights[k].Col+j] = rand()
			}
			net.biasWeights[k].Values[j] = rand()
		}
	}

	return &net
}

// Predict : forward propagation from input to output
func (net *Network) Predict(inputData []float64) *matrix.Matrix {
	// forward propagation
	inputs := &matrix.Matrix{
		Row:    1,
		Col:    len(inputData),
		Values: inputData,
	}
	for i := 0; i < len(net.weights); i++ {
		inputs = matrix.Add(matrix.Dot(inputs, net.weights[i]), matrix.ScalMult(net.biasWeights[i], net.bias))
		inputs = matrix.Apply(inputs, sigmoid)
	}

	return inputs
}

// Train : do a prediction, and backpropagate error
func (net *Network) Train(inputData []float64, targetData []float64) {
	// forward propagation with
	inputs := &matrix.Matrix{
		Row:    1,
		Col:    len(inputData),
		Values: inputData,
	}

	outputs := make([]*matrix.Matrix, len(net.weights))
	deltas := make([]*matrix.Matrix, len(net.weights))

	temp := inputs
	for i := 0; i < len(net.weights); i++ {
		temp = matrix.Add(matrix.Dot(temp, net.weights[i]), matrix.ScalMult(net.biasWeights[i], net.bias))
		temp = matrix.Apply(temp, sigmoid)
		outputs[i] = temp
	}

	result := temp
	// find errors
	target := &matrix.Matrix{
		Row:    1,
		Col:    len(targetData),
		Values: targetData,
	}
	deltas[len(deltas)-1] = matrix.Mult(matrix.Sub(target, result), matrix.Apply(result, dsigmoid))
	for i := len(deltas) - 2; i >= 0; i-- {
		deltas[i] = matrix.Mult(matrix.Dot(deltas[i+1], matrix.Transpose(net.weights[i+1])), matrix.Apply(outputs[i], dsigmoid))
	}
	for i := len(deltas) - 1; i >= 0; i-- {
		if i != 0 {
			net.weights[i] = matrix.Add(net.weights[i], matrix.ScalMult(matrix.Dot(matrix.Transpose(outputs[i-1]), deltas[i]), net.learningRate))
			val := make([]float64, net.weights[i].Row)
			for k := 0; k < net.weights[i].Row; k++ {
				val[k] = 1.0
			}
			ones := &matrix.Matrix{
				Col:    net.weights[i].Row,
				Row:    1,
				Values: val,
			}
			net.biasWeights[i] = matrix.Add(net.biasWeights[i], matrix.ScalMult(matrix.Dot(matrix.Transpose(ones), deltas[i]), net.learningRate))
		} else {
			net.weights[i] = matrix.Add(net.weights[i], matrix.ScalMult(matrix.Dot(matrix.Transpose(inputs), deltas[i]), net.learningRate))
			val := make([]float64, net.weights[i].Row)
			for k := 0; k < net.weights[i].Row; k++ {
				val[k] = 1.0
			}
			ones := &matrix.Matrix{
				Col:    net.weights[i].Row,
				Row:    1,
				Values: val,
			}
			net.biasWeights[i] = matrix.Add(net.biasWeights[i], matrix.ScalMult(matrix.Dot(matrix.Transpose(ones), deltas[i]), net.learningRate))
		}
	}

}

func sigmoid(z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func dsigmoid(z float64) float64 {
	return z * (1 - z)
}

func initArray(size int) []float64 {
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = 1
	}
	return data
}

var seed int = 1103527590

func rand() float64 {
	val := float64(float32(seed) / float32(0x7fffffff))
	seed = (1103515245*seed + 12345) % (0x80000000)
	return val
}
