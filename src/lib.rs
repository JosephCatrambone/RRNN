
extern crate rand;

use std::f32;
use std::ops::*;
use rand::Rng;
// let mut rng = rand::thread_rng();
// rng.gen() or rng.gen::<f32>() or rand::random::<(some tuple of values)>

#[repr(C)]
pub struct Matrix {
	rows : usize, // Height
	columns : usize, // Width
	data : Vec<f32>,
}

impl Matrix {
	fn new(num_rows : usize, num_cols : usize) -> Matrix {
		Matrix {
			rows : num_rows,
			columns : num_cols,
			data : vec![0.0f32; num_rows*num_cols],
		}
	}

	fn new_from_row(d : &Vec<f32>) -> Matrix {
		Matrix { rows : 1, columns : d.len(), data : d.clone() }
	}

	fn new_from_fn(num_rows : usize, num_cols : usize, f : Box<Fn(usize, usize)->f32>) -> Matrix {
		let mut new_mat = Matrix {
			rows : num_rows,
			columns : num_cols,
			data : vec![]
		};

		for i in 0..num_rows {
			for j in 0..num_cols {
				new_mat.data.push(f(i, j));
			}
		}

		new_mat
	}

	fn new_random(num_rows : usize, num_cols : usize, scale : f32) -> Matrix {
		Matrix::new_from_fn(num_rows, num_cols, Box::new(move |i,j| { rand::random::<f32>()*scale }))
	}

	fn get(&self, row_y : usize, col_x : usize) -> f32 {
		self.data[col_x + row_y*self.columns] // x + y*width
	}

	fn set(&mut self, row_y : usize, col_x : usize, value : f32) {
		self.data[col_x + row_y*self.columns] = value;
	}

	fn transpose(&self) -> Matrix {
		let mut new_mat = Matrix {
			rows : self.columns,
			columns : self.rows,
			data : vec![],
		};

		for i in 0..self.columns {
			for j in 0..self.rows {
				new_mat.data.push(self.get(i, j)); // We iterate ROW MAJOR across this and push it for column major access.
			}
		}

		new_mat
	}

	fn transpose_i(&mut self) {
		for i in 0..self.rows {
			for j in i+1..self.columns {
				let temp = self.get(i, j);
				let ji = self.get(j, i);
				{
					self.set(i, j, ji);
					self.set(j, i, temp);
				}
			}
		}
	}

	fn element_unary_op(&self, f : Box<Fn(f32)->f32>) -> Matrix {
		let mut new_mat = Matrix {
			rows : self.rows,
			columns : self.columns,
			data : vec![]
		};

		for i in 0..self.rows {
			for j in 0..self.columns {
				new_mat.data.push(f(self.get(i, j)));
			}
		}

		new_mat
	}

	fn element_unary_op_i(&mut self, f : Box<Fn(f32)->f32>) {
		for i in 0..self.rows {
			for j in 0..self.columns {
				let v = f(self.get(i, j));
				self.set(i, j, v);
			}
		}
	}

	fn element_binary_op(&self, other : &Matrix, f : Box<Fn(f32, f32)->f32>) -> Matrix {
		let mut new_mat = Matrix {
			rows : self.rows,
			columns : self.columns,
			data : vec![]
		};

		for i in 0..self.rows {
			for j in 0..self.columns {
				new_mat.set(i, j, f(self.get(i, j), other.get(i, j)));
			}
		}

		new_mat
	}

	fn element_binary_op_i(&mut self, other : &Matrix, f : Box<Fn(f32, f32)->f32>) {
		for i in 0..self.rows {
			for j in 0..self.columns {
				let v = f(self.get(i, j), other.get(i, j));
				self.set(i, j, v);
			}
		}
	}

	fn multiply(&self, other : &Matrix) -> Matrix {
		assert_eq!(self.columns, other.rows);
		let mut new_mat = Matrix {
			rows : self.rows,
			columns : other.columns,
			data : vec![]
		};

		for i in 0..self.rows {
			for j in 0..other.columns {
				let mut accumulator = 0.0;
				for k in 0..self.columns {
					accumulator += self.get(i, k)*other.get(k, j);
				}
				new_mat.data.push(accumulator);
			}
		}

		new_mat
	}
}

/*
impl Add for Matrix {
	type Output = Matrix;

	fn add(self, _rhs : Matrix) -> Matrix {
	}
}
*/

pub struct RNN {
	weight_ih : Matrix,
	weight_hh : Matrix,
	weight_ho : Matrix,
	bias_h : Matrix,
	bias_o : Matrix,

	hidden_state : Matrix,
}

impl RNN {
	fn new(input_size : usize, hidden_size : usize) -> RNN {
		let mut rng = rand::thread_rng();
		let scale : f32 = 0.1;
		//let random_constructor = Box::new(|i,j| { let mut rng = rand::thread_rng(); rng.gen::<f32>()*0.1 });
		RNN {
			weight_ih : Matrix::new_random(input_size, hidden_size, 0.1f32),
			weight_hh : Matrix::new_random(hidden_size, hidden_size, 0.1f32), 
			weight_ho : Matrix::new_random(hidden_size, input_size, 0.1f32),
			//weight_ho : Matrix { rows : hidden_size, columns : input_size, data : (0..hidden_size*input_size).into_iter().map(|x| { rng.gen::<f32>()*scale }).collect() },
			bias_h : Matrix::new_random(1, hidden_size, 0.1f32),
			bias_o : Matrix::new_random(1, input_size, 0.1f32),
			hidden_state : Matrix { rows: 1, columns : hidden_size, data : vec![0.0; hidden_size] }
		}
	}

	fn reset_hidden_state(&mut self) {
		for i in 0..self.hidden_state.data.len() {
			self.hidden_state.data[i] = 0.0;
		}
	}

	fn set_hidden_state(&mut self, state : Vec<f32>) {
		self.hidden_state.data.copy_from_slice(&state)
	}

	fn get_hidden_state(&self) -> Vec<f32> {
		self.hidden_state.data.clone()
	}

	fn step(&mut self, input_example : &Vec<f32>) -> Vec<f32> {
		let input_mat = Matrix::new_from_row(input_example);

		// Multiply hidden activity through the system.
		let mut new_hidden_accumulator = input_mat.multiply(&self.weight_ih); // x * Wih
		new_hidden_accumulator.element_binary_op_i(&self.hidden_state.multiply(&self.weight_hh), Box::new(|a, b|{a+b})); // + h * Whh
		new_hidden_accumulator.element_binary_op_i(&self.bias_h, Box::new(|a, b|{a+b})); // + hb
		new_hidden_accumulator.element_unary_op_i(Box::new(|a|{ a.tanh() })); // tanh(x*Wih + h*Whh + hb)

		// Copy hidden data back into the hidden state.
		// self.hidden_state.data = new_hidden_accumulator.data, maybe?
		self.hidden_state.data.copy_from_slice(&new_hidden_accumulator.data);
		
		// Multiply for new output.
		let mut result = self.hidden_state.multiply(&self.weight_ho);
		result.element_binary_op_i(&self.bias_o, Box::new(|a, b|{a+b}));
		result.data
	}
}

fn loss_function(rnn : &mut RNN, example : &Vec<f32>, target : &Vec<f32>) {

	rnn.reset_hidden_state();

	// Forward pass.
	/*
		xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
		xs[t][inputs[t]] = 1
		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
		ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
		ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
		loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
	*/
	let mut output = rnn.step(&example); // Gives us ys.
	let error : Vec<f32> = target.into_iter().zip(output.into_iter()).map(|d| { (d.1 - d.0) }).collect(); // Squared error?
	let loss : f32 = error.into_iter().fold(0.0, |sum, difference| { sum + difference.powi(2) }); // Fold squared error together.
	panic!("Start here.");

	// Backwards pass.
	/*
		# backward pass: compute gradients going backwards
		dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
		dbh, dby = np.zeros_like(bh), np.zeros_like(by)
		dhnext = np.zeros_like(hs[0])
		for t in reversed(xrange(len(inputs))):
			dy = np.copy(ps[t])
			dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
			dWhy += np.dot(dy, hs[t].T)
			dby += dy
			dh = np.dot(Why.T, dy) + dhnext # backprop into h
			dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(Whh.T, dhraw)
		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
	*/
	
}

pub fn train_rnn(rnn : &mut RNN, iterations : usize, training_data : &[Vec<f32>], label_data : &[Vec<f32>]) {
	
}

#[no_mangle]
pub extern "C" fn test_method(foo : &str) {
	println!("Got string: {}", foo);
}

//#[no_mangle]
//pub extern "C" fn load_rnn(filename : &str)

#[cfg(test)]
mod tests {
	use super::{Matrix, test_method, RNN};

	#[test]
	fn test_ident() {
		//test_method("Foooo!");
		// Verify that the ident behaves as we expect.
		let ident = Matrix { rows : 3, columns : 3, data : vec![1.0, 0.0, 0.0,   0.0, 1.0, 0.0,   0.0, 0.0, 1.0] };
		let ident_maybe = Matrix::new_from_fn(3, 3, Box::new(|i,j| { if i==j { 1.0 } else { 0.0 }}));
		assert!(ident.data == ident_maybe.data);

		let step = Matrix { rows : 3, columns : 4, data : (0..12u32).into_iter().map(|x| { x as f32 }).collect() };
		let result = ident.multiply(&step);
		assert!(result.data == step.data);
	}

	#[test]
	fn test_matmul1() {
		let mut a = Matrix::new(3, 4);
		let mut b = Matrix::new(4, 5);

		a.element_unary_op_i(Box::new(|x| { 1.0f32 })); // set all A to 1.
		b.element_unary_op_i(Box::new(|x| { 1.0f32 })); // Set all B to 1.

		let c = a.multiply(&b);

		assert_eq!(c.rows, 3);
		assert_eq!(c.columns, 5);
		for i in 0..3*5 {
			assert_eq!(c.data[i], 4.0); // ones * ones should all be 4.
		}
	}
}
