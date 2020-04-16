package matrix

import (
	"testing"
)

func TestMatrix(t *testing.T) {
	values := make([]float64, 4)
	matrix := &Matrix{Row: 2, Col: 2, Values: values}
	if matrix.Row != 2 {
		t.Errorf("Matrix.row = %d; want 2", matrix.Row)
	}
	if matrix.Col != 2 {
		t.Errorf("Matrix.col = %d; want 2", matrix.Col)
	}
	for i, v := range matrix.Values {
		if v != 0 {
			t.Errorf("Matrix.values[%d] = %f; want 0", i, v)
		}
	}
}

func TestScalAdd(t *testing.T) {
	values := []float64{0, 1, 2, 3}
	matrix := &Matrix{Row: 2, Col: 2, Values: values}

	scal := float64(8)
	expectedValues := []float64{8, 9, 10, 11}
	matrix1 := ScalAdd(matrix, scal)
	for i, v := range matrix1.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestMatrixAdd(t *testing.T) {
	values1 := []float64{0, 1, 2, 3}
	matrix1 := &Matrix{Row: 2, Col: 2, Values: values1}

	values2 := []float64{4, 8, 3, 5}
	matrix2 := &Matrix{Row: 2, Col: 2, Values: values2}

	expectedValues := []float64{4, 9, 5, 8}

	matrix := Add(matrix1, matrix2)

	for i, v := range matrix.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestMatrixMult(t *testing.T) {
	values1 := []float64{0, 1, 2, 3}
	matrix1 := &Matrix{Row: 2, Col: 2, Values: values1}

	values2 := []float64{4, 8, 3, 5}
	matrix2 := &Matrix{Row: 2, Col: 2, Values: values2}

	expectedValues := []float64{0, 8, 6, 15}

	matrix := Mult(matrix1, matrix2)

	for i, v := range matrix.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestMatrixSub(t *testing.T) {
	values1 := []float64{0, 1, 2, 3}
	matrix1 := &Matrix{Row: 2, Col: 2, Values: values1}

	values2 := []float64{4, 8, 3, 5}
	matrix2 := &Matrix{Row: 2, Col: 2, Values: values2}

	expectedValues := []float64{-4, -7, -1, -2}

	matrix := Sub(matrix1, matrix2)

	for i, v := range matrix.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestScalMult(t *testing.T) {
	values := []float64{0, 1, 2, 3}
	matrix := &Matrix{Row: 2, Col: 2, Values: values}

	scal := float64(8)
	expectedValues := []float64{0, 8, 16, 24}
	matrix1 := ScalMult(matrix, scal)
	for i, v := range matrix1.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestMatrixDot(t *testing.T) {
	values1 := []float64{0, 1, 2, 3, 4, 5}
	matrix1 := &Matrix{Row: 3, Col: 2, Values: values1}

	values2 := []float64{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
	matrix2 := &Matrix{Row: 2, Col: 6, Values: values2}

	expectedValues := []float64{8, 9, 10, 11, 12, 13, 28, 33, 38, 43, 48, 53, 48, 57, 66, 75, 84, 93}

	matrix := Dot(matrix1, matrix2)

	for i, v := range matrix.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestTranspose(t *testing.T) {
	values := []float64{1, 2, 3, 0, -6, 7}
	matrix := &Matrix{Row: 3, Col: 2, Values: values}

	expectedValues := []float64{1, 0, 2, -6, 3, 7}
	matrix1 := Transpose(matrix)
	for i, v := range matrix1.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

func TestTrace(t *testing.T) {
	values := []float64{1, 2, 3, -5}
	matrix := &Matrix{Row: 2, Col: 2, Values: values}

	expectedVal := float64(-4)
	v := Trace(matrix)
	if v != expectedVal {
		t.Errorf("Trace(Matrix) = %f; want %f", v, expectedVal)
	}
}

func TestApply(t *testing.T) {
	values := []float64{0, 1, 2, 3}
	matrix := &Matrix{Row: 2, Col: 2, Values: values}

	double := func(v float64) float64 {
		return v * 2
	}

	expectedValues := []float64{0, 2, 4, 6}
	matrix1 := Apply(matrix, double)
	for i, v := range matrix1.Values {
		if v != expectedValues[i] {
			t.Errorf("Matrix.values[%d] = %f; want %f", i, v, expectedValues[i])
		}
	}
}

const COL = 150
const ROW = 150

func BenchmarkMatrixAdd(b *testing.B) {

	values1 := make([]float64, COL*ROW)
	for i := 0; i < COL*ROW; i++ {
		values1[i] = float64(i)
	}
	matrix1 := &Matrix{Row: ROW, Col: COL, Values: values1}

	values2 := make([]float64, COL*ROW)
	for i := 0; i < COL*ROW; i++ {
		values2[i] = float64(i)
	}
	matrix2 := &Matrix{Row: ROW, Col: COL, Values: values2}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Add(matrix1, matrix2)
	}
}

func BenchmarkMatrixDot(b *testing.B) {

	values1 := make([]float64, COL*ROW)
	for i := 0; i < COL*ROW; i++ {
		values1[i] = float64(i)
	}
	matrix1 := &Matrix{Row: ROW, Col: COL, Values: values1}

	values2 := make([]float64, COL*ROW)
	for i := 0; i < COL*ROW; i++ {
		values2[i] = float64(i)
	}
	matrix2 := &Matrix{Row: COL, Col: ROW, Values: values2}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dot(matrix1, matrix2)
	}
}
