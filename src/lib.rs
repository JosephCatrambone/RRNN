
extern crate libc;
extern crate rand;

use libc::{c_char, c_float};
use std::f32;
use std::ffi::CStr;
use std::ops::*;
use rand::Rng;
// let mut rng = rand::thread_rng();
// rng.gen() or rng.gen::<f32>() or rand::random::<(some tuple of values)>

//
// Matrix
//
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
		Matrix::new_from_fn(num_rows, num_cols, Box::new(move |_,_| { rand::random::<f32>()*scale }))
	}

	fn get(&self, row_y : usize, col_x : usize) -> f32 {
		assert!(row_y >= 0 && row_y < self.rows && col_x >= 0 && col_x < self.columns);
		self.data[col_x + row_y*self.columns] // x + y*width
	}

	fn set(&mut self, row_y : usize, col_x : usize, value : f32) {
		assert!(row_y >= 0 && row_y < self.rows && col_x >= 0 && col_x < self.columns);
		self.data[col_x + row_y*self.columns] = value;
	}

	fn transpose(&self) -> Matrix {
		let mut new_mat = Matrix {
			rows : self.columns,
			columns : self.rows,
			data : vec![],
		};

		for j in 0..self.columns {
			for i in 0..self.rows {
				new_mat.data.push(self.get(i, j)); // We iterate ROW MAJOR across this and push it for column major access.
			}
		}

		new_mat
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
		assert_eq!(self.rows, other.rows);
		assert_eq!(self.columns, other.columns);

		let mut new_mat = Matrix {
			rows : self.rows,
			columns : self.columns,
			data : vec![]
		};

		for i in 0..self.rows {
			for j in 0..self.columns {
				new_mat.data.push(f(self.get(i, j), other.get(i, j)));
			}
		}

		new_mat
	}

	fn element_binary_op_i(&mut self, other : &Matrix, f : Box<Fn(f32, f32)->f32>) {
		assert_eq!(self.rows, other.rows);
		assert_eq!(self.columns, other.columns);

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

//
// RNN
//
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
		let scale : f32 = 0.1;
		//let random_constructor = Box::new(|i,j| { let mut rng = rand::thread_rng(); rng.gen::<f32>()*0.1 });
		RNN {
			weight_ih : Matrix::new_random(input_size, hidden_size, scale),
			weight_hh : Matrix::new_random(hidden_size, hidden_size, scale), 
			weight_ho : Matrix::new_random(hidden_size, input_size, scale),
			//weight_ho : Matrix { rows : hidden_size, columns : input_size, data : (0..hidden_size*input_size).into_iter().map(|x| { rng.gen::<f32>()*scale }).collect() },
			bias_h : Matrix::new_random(1, hidden_size, scale),
			bias_o : Matrix::new_random(1, input_size, scale),
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

/*
impl Iterator for RNN {
	type Item = Vec<f32>;

	fn next(&mut self) -> Option<Vec<f32>> {
	}
}
*/

fn loss_function(rnn : &mut RNN, example : &Vec<f32>, target : &Vec<f32>) -> (f32, Matrix, Matrix, Matrix, Matrix, Matrix) {
	// NOTE: No reset here.
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
	let error : Vec<f32> = target.into_iter().zip(output.into_iter()).map(|d| { d.1 - d.0 }).collect(); // Want this to be how far from the truth we are.  If out was [0.1, 0.9] and truth was [0.0, 1.0], this would be [0.1, -0.1]
	let loss : f32 = error.clone().into_iter().fold(0.0, |sum, difference| { sum + difference.powi(2) }); // Fold squared error together.

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
	//println!("RNN State shapes: Wih: {} {}", rnn.weight_ih.rows, rnn.weight_ih.columns);
	//println!("RNN State shapes: Whh: {} {}", rnn.weight_hh.rows, rnn.weight_hh.columns);
	//println!("RNN State shapes: Who: {} {}", rnn.weight_ho.rows, rnn.weight_ho.columns);

	let delta_y = Matrix::new_from_row(&error);
	let delta_bias_y = Matrix::new_from_row(&error); // Also delta y.
	//println!("Delta y shape: {} {}", delta_y.rows, delta_y.columns);
	let delta_weight_hy = rnn.hidden_state.transpose().multiply(&delta_y);
	//println!("Delta weight hy: {} {}", delta_weight_hy.rows, delta_weight_hy.columns);
	
	let delta_h = delta_y.multiply(&rnn.weight_ho.transpose());
	//println!("Delta hidden: {} {}", delta_h.rows, delta_h.columns);

	let delta_h_derivative = rnn.hidden_state.element_unary_op(Box::new(|x| { 1.0 - (x*x) })); // Splitting dhraw = (1 - hs[t]*hs[t]) * dh into two lines.
	let delta_h_raw = rnn.hidden_state.element_binary_op(&delta_h, Box::new(|a,b| { a*b })); // TODO: op_i?
	//println!("Delta hidden raw: {} {}", delta_h_raw.rows, delta_h_raw.columns);

	let delta_bias_h = Matrix::new_from_row(&delta_h_raw.data); // Dimensino mismatch?
	let x_in = Matrix::new_from_row(&example);
	//println!("Dims for multiply: {}x{} and {}x{}", delta_h_raw.rows, delta_h_raw.columns, x_in.rows, x_in.columns);
	let delta_weight_xh = x_in.transpose().multiply(&delta_h_raw);
	//println!("Delta weight xh: {} {}", delta_weight_xh.rows, delta_weight_xh.columns);

	let delta_weight_hh = rnn.hidden_state.transpose().multiply(&delta_h_raw);

	// TODO: Clip weights.
	return (loss, delta_weight_xh, delta_weight_hh, delta_weight_hy, delta_bias_h, delta_bias_y);
}

pub fn train_sequence(rnn : &mut RNN, learning_rate : f32, training_data : &[Vec<f32>], label_data : &[Vec<f32>], verbose : bool) {
	let lr = learning_rate;
	let mut step = 0;
	rnn.reset_hidden_state();
	for (ex, target) in training_data.iter().zip(label_data.iter()) {
		// Get the loss + errors.
		let (loss, dwih, dwhh, dwho, dbh, dbo) = loss_function(rnn, ex, target);
		if verbose {
			println!("Step {} - Loss {}", step, loss);
		}
		// Apply gradients.
		rnn.weight_ih.element_binary_op_i(&dwih, Box::new(move |a, b|{a - b*lr}));
		rnn.weight_hh.element_binary_op_i(&dwhh, Box::new(move |a, b|{a - b*lr}));
		rnn.weight_ho.element_binary_op_i(&dwho, Box::new(move |a, b|{a - b*lr}));
		rnn.bias_h.element_binary_op_i(&dbh, Box::new(move |a, b|{a - b*lr}));
		rnn.bias_o.element_binary_op_i(&dbo, Box::new(move |a, b|{a - b*lr}));

		step += 1;
	}
	//lr *= learning_decay;
}

//
// BEGIN C API
//
const INPUT_SIZE : usize = 28; // 26 Letters, space, EOL.
const SPACE_CHARACTER : usize = 26;
const TERMINAL_CHARACTER : usize = 27;
const ASCII_a : u8 = 97; // Lower-case is 97.
const ASCII_z : u8 = 122;
const ASCII_SPACE : u8 = 32;

// Helpers.
fn string_to_vector(s : &str) -> Vec<Vec<f32>> {
	let mut result : Vec<Vec<f32>> = vec![];
	for c in s.to_lowercase().as_bytes() {
		let mut char_vector = vec![0.0f32; INPUT_SIZE];
		if *c >= ASCII_a && *c <= ASCII_z {
			char_vector[(c - ASCII_a) as usize] = 1.0;
		} else if *c == ASCII_SPACE {
			char_vector[SPACE_CHARACTER] = 1.0;
		}
		result.push(char_vector);
	}
	// Append the end of line marker.
	let mut terminal_vector = vec![0.0f32; INPUT_SIZE];
	terminal_vector[TERMINAL_CHARACTER] = 1.0;
	result.push(terminal_vector);
	result
}

fn vector_to_string(v : Vec<Vec<f32>>) -> String {
	let mut res = String::new();
	// res.push("asdf")
	for character_distribution in v.iter() {
		// Randomly select from the distribution, assuming uniform.
		let mut energy = rand::random::<f32>();
		// For each of the entries in the distribution...
		for (index, candidate) in character_distribution.iter().enumerate() {
			// If the candidate energy is greater than this bump, lose that energy and continue.
			if energy > *candidate {
				energy -= *candidate;
			} else {
				// We have selected a character.
				if index == SPACE_CHARACTER {
					res.push(' ');
				} else if index == TERMINAL_CHARACTER {
					return res; // Break early.
				} else {
					res.push(std::char::from_u32((ASCII_a as usize + index) as u32).unwrap());
				}
				break;
			}
		}
	}
	res
}

#[no_mangle]
pub extern "C" fn build_rnn(hidden_layer_size : usize) -> *mut RNN {
	// Train our RNN.
	let mut rnn = RNN::new(INPUT_SIZE, hidden_layer_size);
	
	Box::into_raw(Box::new(rnn))
}

#[no_mangle]
pub extern "C" fn train_rnn_with_corpus(rnn_ptr : *mut RNN, corpus : *mut c_char, delimiter : char, iterations : u32, learning_rate : f32, learning_decay : f32, verbose : bool) {
	let mut lr = learning_rate;

	let mut rnn = unsafe {
		assert!(!rnn_ptr.is_null());
		&mut *rnn_ptr
	};

	// Split up our corpus.
	let strings : &str = unsafe {
		assert!(!corpus.is_null());
		CStr::from_ptr(corpus) // use from_raw(corpus) if CString
	}.to_str().unwrap();

	//"foo\rbar\nbaz\rquux".split('\n').flat_map(|x| x.split('\r')).collect::<Vec<_>>()	
	for _ in 0..iterations {
		for s in strings.split(delimiter) {
			let training_sample = string_to_vector(&s);
			// Convert Vec<Vec<f32>> to &[Vec<f32>] for learning..
			train_sequence(&mut rnn, lr, &training_sample[..], &training_sample[1..], verbose);
		}
		lr *= learning_decay;
	}
}

#[no_mangle]
pub extern "C" fn transform_document(rnn_ptr : *mut RNN, document : *mut c_char, reset_rnn : bool) -> *mut c_float {
	let mut rnn = unsafe {
		assert!(!rnn_ptr.is_null());
		&mut *rnn_ptr
	};

	let doc = unsafe {
		assert!(!document.is_null());
		//CString::from_raw(document) // Or CStr, but we dont' want &str
		CStr::from_ptr(document)
	}.to_str().unwrap();

	// Reset RNN.
	if reset_rnn  {
		rnn.reset_hidden_state();
	}

	// Process string.
	let doc_vector = string_to_vector(&doc);
	for v in doc_vector {
		rnn.step(&v);
	}

	// Return hidden state.
	rnn.hidden_state.data.clone().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn load_rnn(filename : &str) -> *mut RNN {
	//let serialized = serde_json::to_string(&point).unwrap();
	//println!("serialized = {}", serialized);
	let new_rnn = RNN::new(1,1);
	Box::into_raw(Box::new(new_rnn))
}

#[no_mangle]
pub extern "C" fn free_rnn(rnn_ptr : *mut RNN) {
	let rnn = unsafe {
		if rnn_ptr.is_null() { return; }
		&*rnn_ptr
	};
}

//
// TESTS
//
#[cfg(test)]
mod tests {
	use super::{Matrix, RNN, train_sequence, vector_to_string, string_to_vector};
	//use super::*;

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

		a.element_unary_op_i(Box::new(|_| { 1.0f32 })); // set all A to 1.
		b.element_unary_op_i(Box::new(|_| { 1.0f32 })); // Set all B to 1.

		let c = a.multiply(&b);

		assert_eq!(c.rows, 3);
		assert_eq!(c.columns, 5);
		for i in 0..3*5 {
			assert_eq!(c.data[i], 4.0); // ones * ones should all be 4.
		}
	}

	#[test]
	#[cfg_attr(not(feature="expensive_tests"), ignore)] // Run with `cargo test --features expensive_tests`
	fn test_train_blink_sequence() {
		let mut rnn = RNN::new(3, 2);
		for _ in 0..10000 {
			train_sequence(&mut rnn, 0.01, &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]], &[vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0]], true);
		}
		//loss_function(&mut rnn, &vec![1.0, 0.0, 0.0], &vec![0.0, 1.0, 0.0]);
		rnn.reset_hidden_state();
		let next = rnn.step(&vec![1.0, 0.0, 0.0]);
		println!("Next: {:?}", next);
		assert!(next[0] < 0.1 && next[1] > 0.9 && next[2] < 0.1);
	}

	#[test]
	fn test_vector_conversion_sanity() {
		assert_eq!("foo bar bez bum", vector_to_string(string_to_vector("foo bar bez bum")));
	}
}
