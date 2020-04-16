package matrix

// Matrix : a matrix
type Matrix struct {
	Row, Col int
	Values   []float64
}

// ScalAdd : return a matrix by adding scal element wise
func ScalAdd(m *Matrix, scal float64) *Matrix {
	values := make([]float64, m.Col*m.Row)
	for i := 0; i < m.Row*m.Col; i++ {
		values[i] = m.Values[i] + scal
	}
	return &Matrix{Row: m.Row, Col: m.Col, Values: values}
}

// Add : return a matrix from the addition of two matrixes element wise
func Add(m1 *Matrix, m2 *Matrix) *Matrix {
	values := make([]float64, m1.Col*m1.Row)
	for i := 0; i < m1.Row*m1.Col; i++ {
		val1 := m1.Values[i]
		val2 := m2.Values[i]
		val := val1 + val2
		values[i] = val
	}
	return &Matrix{Row: m1.Row, Col: m1.Col, Values: values}
}

// Sub : return a matrix from the substraction of two matrixes element wise
func Sub(m1 *Matrix, m2 *Matrix) *Matrix {
	values := make([]float64, m1.Col*m1.Row)
	for i := 0; i < m1.Row*m1.Col; i++ {
		val1 := m1.Values[i]
		val2 := m2.Values[i]
		val := val1 - val2
		values[i] = val
	}
	return &Matrix{Row: m1.Row, Col: m1.Col, Values: values}
}

// ScalMult : return a matrix by multiplying element wise by scal
func ScalMult(m *Matrix, scal float64) *Matrix {
	values := make([]float64, m.Col*m.Row)
	for i := 0; i < m.Row*m.Col; i++ {
		values[i] = m.Values[i] * scal
	}
	return &Matrix{Row: m.Row, Col: m.Col, Values: values}
}

// Transpose : return a matrix by transposition
func Transpose(m *Matrix) *Matrix {
	values := make([]float64, m.Col*m.Row)
	for i := 0; i < m.Row; i++ {
		for j := 0; j < m.Col; j++ {
			values[i*m.Col+j] = m.Values[j*m.Row+i]
		}
	}
	return &Matrix{Row: m.Col, Col: m.Row, Values: values}
}

// Mult : return a matrix from the multiplication of two matrixes element wise
func Mult(m1 *Matrix, m2 *Matrix) *Matrix {
	values := make([]float64, m1.Col*m1.Row)
	for i := 0; i < m1.Row*m1.Col; i++ {
		val1 := m1.Values[i]
		val2 := m2.Values[i]
		val := val1 * val2
		values[i] = val
	}
	return &Matrix{Row: m1.Row, Col: m1.Col, Values: values}
}

// Dot : return a matrix by multiplying two Matrixes
func Dot(m1 *Matrix, m2 *Matrix) *Matrix {
	values := make([]float64, m1.Row*m2.Col)
	for i := 0; i < m1.Row; i++ {
		for j := 0; j < m2.Col; j++ {
			for k := 0; k < m1.Col; k++ {
				values[i*m2.Col+j] += m1.Values[i*m1.Col+k] * m2.Values[k*m2.Col+j]
			}
		}
	}
	return &Matrix{Row: m1.Row, Col: m2.Col, Values: values}
}

// Trace : return the trace of a square matrix
func Trace(m *Matrix) float64 {
	value := float64(0)
	for i := 0; i < m.Row; i++ {
		value += m.Values[i*m.Col+i]
	}
	return value
}

// Apply : return a matrix by applying function element wise
func Apply(m *Matrix, fn func(v float64) float64) *Matrix {
	values := make([]float64, m.Col*m.Row)
	for i := 0; i < m.Row*m.Col; i++ {
		values[i] = fn(m.Values[i])
	}
	return &Matrix{Row: m.Row, Col: m.Col, Values: values}
}
