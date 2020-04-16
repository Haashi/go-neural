package network

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/haashi/go-neural/matrix"
)

func TestCreateNetwork(t *testing.T) {
	type args struct {
		inputs       int
		hiddens      int
		layersSize   []int
		outputs      int
		learningRate float64
		bias         float64
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "1",
			args: args{inputs: 2, hiddens: 2, layersSize: []int{3, 4}, learningRate: 0.1, outputs: 1, bias: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			CreateNetwork(tt.args.inputs, tt.args.hiddens, tt.args.layersSize, tt.args.outputs, tt.args.learningRate, tt.args.bias)
		})
	}
}

func TestNetwork_Predict(t *testing.T) {
	type fields struct {
		inputs       int
		hiddenLayers int
		outputs      int
		weights      []*matrix.Matrix
		biasWeights  []*matrix.Matrix
		learningRate float64
		bias         float64
	}
	type args struct {
		inputData []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *matrix.Matrix
	}{
		{
			name: "1",
			args: args{
				inputData: []float64{1, 1},
			},
			fields: fields{bias: 1, inputs: 2, hiddenLayers: 2, learningRate: 0.1, outputs: 1, weights: []*matrix.Matrix{
				{Col: 2, Row: 2, Values: []float64{
					2, 0,
					1, 4,
				}},
				{Col: 2, Row: 2, Values: []float64{
					1, 0,
					1, 1,
				}},
				{Col: 1, Row: 2, Values: []float64{
					1,
					0,
				}},
			}, biasWeights: []*matrix.Matrix{
				{Col: 2, Row: 1, Values: []float64{
					1, 0,
				}},
				{Col: 2, Row: 1, Values: []float64{
					0, 1,
				}},
				{Col: 1, Row: 1, Values: []float64{
					1,
				}},
			},
			},
			want: &matrix.Matrix{Col: 1, Row: 1, Values: []float64{sigmoid(sigmoid(2*sigmoid(4)) + 1)}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			net := &Network{
				inputs:       tt.fields.inputs,
				hiddenLayers: tt.fields.hiddenLayers,
				outputs:      tt.fields.outputs,
				weights:      tt.fields.weights,
				learningRate: tt.fields.learningRate,
				bias:         tt.fields.bias,
				biasWeights:  tt.fields.biasWeights,
			}
			if got := net.Predict(tt.args.inputData); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Network.Predict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNetwork_Train(t *testing.T) {
	type fields struct {
		inputs       int
		hiddenLayers int
		outputs      int
		weights      []*matrix.Matrix
		biasWeights  []*matrix.Matrix
		learningRate float64
		bias         float64
	}
	type args struct {
		inputData  []float64
		targetData []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "1",
			args: args{
				inputData:  []float64{0},
				targetData: []float64{0},
			},
			fields: fields{bias: 1, inputs: 1, hiddenLayers: 0, learningRate: 0.5, outputs: 1, weights: []*matrix.Matrix{
				{Col: 1, Row: 1, Values: []float64{
					0.17,
				}},
			}, biasWeights: []*matrix.Matrix{
				{Col: 1, Row: 1, Values: []float64{
					0.30,
				}},
			},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			net := &Network{
				inputs:       tt.fields.inputs,
				hiddenLayers: tt.fields.hiddenLayers,
				outputs:      tt.fields.outputs,
				weights:      tt.fields.weights,
				learningRate: tt.fields.learningRate,
				bias:         tt.fields.bias,
				biasWeights:  tt.fields.biasWeights,
			}
			net.Train(tt.args.inputData, tt.args.targetData)
		})
	}
}

func TestNetwork_RealCase(t *testing.T) {
	type args struct {
		inputs       int
		hiddenLayers int
		outputs      int
		layersSize   []int
		learningRate float64
		bias         float64
	}
	type training struct {
		input  []float64
		target []float64
	}
	tests := []struct {
		name         string
		args         args
		trainingData []training
	}{
		{
			name: "copycat",
			trainingData: []training{
				{input: []float64{0}, target: []float64{0}},
				{input: []float64{1}, target: []float64{1}},
			},
			args: args{bias: 1, inputs: 1, hiddenLayers: 0, learningRate: 0.5, outputs: 1, layersSize: []int{}},
		},
		{
			name: "Or",
			trainingData: []training{
				{input: []float64{0, 0}, target: []float64{0}},
				{input: []float64{1, 1}, target: []float64{1}},
				{input: []float64{0, 1}, target: []float64{1}},
				{input: []float64{1, 0}, target: []float64{1}},
			},
			args: args{bias: 1, inputs: 2, hiddenLayers: 0, learningRate: 0.5, outputs: 1, layersSize: []int{}},
		},
		{
			name: "2 HIDDEN LAYERS",
			trainingData: []training{
				{input: []float64{0, 0, 0, 0}, target: []float64{1}},
				{input: []float64{0, 0, 0, 1}, target: []float64{0}},
				{input: []float64{0, 0, 1, 0}, target: []float64{0}},
				{input: []float64{0, 0, 1, 1}, target: []float64{0}},
				{input: []float64{0, 1, 0, 0}, target: []float64{0}},
				{input: []float64{0, 1, 0, 1}, target: []float64{1}},
				{input: []float64{0, 1, 1, 0}, target: []float64{0}},
				{input: []float64{0, 1, 1, 1}, target: []float64{0}},
				{input: []float64{1, 0, 0, 0}, target: []float64{0}},
				{input: []float64{1, 0, 0, 1}, target: []float64{0}},
				{input: []float64{1, 0, 1, 0}, target: []float64{1}},
				{input: []float64{1, 0, 1, 1}, target: []float64{0}},
				{input: []float64{1, 1, 0, 0}, target: []float64{0}},
				{input: []float64{1, 1, 0, 1}, target: []float64{0}},
				{input: []float64{1, 1, 1, 0}, target: []float64{0}},
				{input: []float64{1, 1, 1, 1}, target: []float64{1}},
			},
			args: args{bias: 1, inputs: 4, hiddenLayers: 2, learningRate: 0.5, outputs: 1, layersSize: []int{2, 2}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			net := CreateNetwork(tt.args.inputs, tt.args.hiddenLayers, tt.args.layersSize, tt.args.outputs, tt.args.learningRate, tt.args.bias)
			for i := 0; i < 10000; i++ {
				for _, test := range tt.trainingData {
					net.Train(test.input, test.target)
				}
			}
			for k, weight := range net.weights {
				fmt.Fprintf(os.Stderr, "weight layer n°%d : %+v\n", k, weight)
			}
			for k, weight := range net.biasWeights {
				fmt.Fprintf(os.Stderr, "bias layer n°%d : %+v\n", k, weight)
			}
			for _, test := range tt.trainingData {
				val := net.Predict(test.input)
				for i := range val.Values {
					val.Values[i] = math.Round(val.Values[i])
					if !reflect.DeepEqual(val.Values, test.target) {
						t.Errorf("Network.Predict(%v) = %v, want %v", test.input, val.Values[i], test.target)
					}
				}
			}
		})
	}
}
