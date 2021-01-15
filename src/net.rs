use rand::prelude::*;
use rand::distributions::Uniform;

pub type Output = [f64; 10];

#[link(name="operations")]
extern {
	fn convolute28(input: *const f64, out: *mut f64, conv: *const f64);
	fn convoluted28_backprop(input: *const f64, out: *const f64,
		                     d_conv: *mut f64);
	fn pool26(input: *const f64, out: *mut f64);
	fn convolute13(input: *const f64, out: *mut f64, conv: *const f64);
	fn convoluted13_backprop(input: *const f64, out: *const f64,
		                     d_conv: *mut f64);
	fn convoluted13_backprop_input(input: *const f64,
		                           out: *const f64,
		                           id_conv: *mut f64);
	fn pool11(input: *const f64, out: *mut f64);
	fn pool11_backprop(input: *const f64, out: *const f64,
		              back: *mut f64, d_loss: *const f64) -> i32;
	fn pool26_backprop(input: *const f64, out: *const f64,
		              back: *mut f64, d_loss: *const f64) -> i32;
}

pub struct Net {
	// Nodes
	// Layer1 is a convolution and a bias
	layer1_conv_weights: Vec<f64>,
	layer1_bias_weights: Vec<f64>,

	// Layer 2 is a convolution + bias
	layer2_conv_weights: Vec<f64>,
	layer2_bias_weights: Vec<f64>,

	// Layer 3 is a deep layer
	layer3_weights: Vec<f64>,
	layer3_bias_weights: Vec<f64>,

	// output layer
	output_weights: Vec<f64>,
	output_bias_weights: Vec<f64>,

	// Buffers
	input_image: Vec<f64>,
	layer1_conv: Vec<f64>,
	layer1_conv_bias_relu: Vec<f64>,
	layer1_pooled: Vec<f64>,

	layer2_conv: Vec<f64>,
	layer2_conv_bias_relu: Vec<f64>,
	layer2_pooled: Vec<f64>,

	layer3_out: Vec<f64>,

	output_layer: Vec<f64>,
	output_layer_bias_exp: Vec<f64>,

	// train int
	train_number: u64,
}

impl Net {
	pub fn new() -> Self {
	let mut layer1_conv_weights = vec![0.0; 9 * 32];
	let mut layer1_bias_weights  = vec![0.0; 32];

	// Layer 2 is a convolution + bias
	let mut layer2_conv_weights = vec![0.0; 9 * 64];
	let mut layer2_bias_weights = vec![0.0; 64];

	// Layer 3 is a deep layer
	let mut layer3_weights = vec![0.0; 5 * 5 * 32 * 64 * 64];
	let mut layer3_bias_weights = vec![0.0; 64];

	// output layer
	let mut output_weights = vec![0.0; 64 * 10];
	let mut output_bias_weights = vec![0.0; 10];


	layer1_conv_weights.iter_mut()
		.chain(layer1_bias_weights.iter_mut())
		.chain(layer2_conv_weights.iter_mut())
		.chain(layer2_bias_weights.iter_mut())
		.chain(layer3_weights.iter_mut())
		.chain(layer3_bias_weights.iter_mut())
		.chain(output_weights.iter_mut())
		.chain(output_bias_weights.iter_mut())
		.for_each(|a| *a = dist());

	// Buffers
	let input_image = vec![0.0; 28 * 28];
	let layer1_conv = vec![0.0; 26 * 26 * 32];
	let layer1_conv_bias_relu = vec![0.0; 26 * 26 * 32];
	let layer1_pooled = vec![0.0; 13 * 13 * 32];

	let layer2_conv = vec![0.0; 11 * 11 * 32 * 64];
	let layer2_conv_bias_relu = vec![0.0; 11 * 11 * 32 * 64];
	let layer2_pooled = vec![0.0; 5 * 5 * 32 * 64];

	let layer3_out = vec![0.0; 64];

	let output_layer = vec![0.0; 10];
	let output_layer_bias_exp = vec![0.0; 10];

		Self{
			layer1_conv_weights: layer1_conv_weights,
			layer1_bias_weights: layer1_bias_weights,
			layer2_conv_weights: layer2_conv_weights,
			layer2_bias_weights: layer2_bias_weights,
			layer3_weights: layer3_weights,
			layer3_bias_weights: layer3_bias_weights,
			output_weights: output_weights,
			output_bias_weights: output_bias_weights,
			input_image: input_image,
			layer1_conv: layer1_conv,
			layer1_conv_bias_relu: layer1_conv_bias_relu,
			layer1_pooled: layer1_pooled,
			layer2_conv: layer2_conv,
			layer2_conv_bias_relu: layer2_conv_bias_relu,
			layer2_pooled: layer2_pooled,
			layer3_out: layer3_out,
			output_layer: output_layer,
			output_layer_bias_exp: output_layer_bias_exp,
			train_number: 0,
		}
	}

	pub fn compute(&mut self,
		           input: &[[f64; 28]; 28],
		           out: &mut Output) {

		// Save the input image
		self.input_image = input
			.iter()
			.flatten()
			.map(|&a| a)
			.collect();

		self.compute_layer1_conv();
		self.compute_layer1_maxpool();
		self.compute_layer2_conv();
		self.compute_layer2_maxpool();
		self.compute_layer3();
		self.compute_output_layer(out);
	}

	pub fn train(&mut self, label: usize, out: &Output,
			     learning_rate: f64)
	{
		let d_loss = 1.0 / out[label];
		self.train_number += 1;

		let d_l_d_i = self.train_output_layer(d_loss, label,
			// Set the learning rate to 0 if we want
			// to do a dropout.
			if self.train_number % 2 == 0 {
				0.0
			} else {
				0.01 * learning_rate
			}
		);

		let d_l_d_i = self.train_layer3(&d_l_d_i,
			learning_rate * 10.0);

		if self.train_number < 2048 {
			return;
		}

		let d_l_d_i = self.train_layer2_maxpool(&d_l_d_i);
		let d_l_d_i = self.train_layer2_conv(&d_l_d_i,
			if self.train_number % 2 != 0 {
				0.0
			} else {
				0.01 * learning_rate
			}
		);

		if self.train_number < 4096 {
			return;
		}

		let mut d_l_d_i = self.train_layer1_maxpool(&d_l_d_i);
		self.train_layer1_conv(&mut d_l_d_i,
			if self.train_number % 2 == 0{
				0.0
			} else {
				learning_rate * 0.001
			}
		);
	}

	fn compute_layer1_conv(&mut self)
	{
		for (node_num, &bias) in (0..32)
			.zip(self.layer1_bias_weights.iter())
		{
			let conv_ind = node_num * 9;
			let conv = &self.layer1_conv_weights[conv_ind..conv_ind+9];
			// 676 is 26^2
			let out_ind = node_num * 676;
			let out = &mut self.layer1_conv[out_ind..out_ind+676];

			// Perform the convolution
			do_convolute28(&self.input_image, out, conv);

			// Apply the bias and the relu
			let out_bias_relu = &mut self.layer1_conv_bias_relu[
				out_ind..out_ind+676
			];
			out.iter()
				.zip(out_bias_relu.iter_mut())
				.for_each(|(&a, b)| *b = leaky_relu(a + bias));
		}
	}

	fn train_layer1_conv(&mut self,
		                 d_l_d_i: &mut [f64],
		                 learning_rate: f64)
	{
		assert_eq!(d_l_d_i.len(), 26 * 26 * 32);

		// Step back through the leaky_relu
		d_l_d_i.iter_mut()
			.for_each(|a| *a = if *a < 0.0 {
				*a * 0.01
			} else {
				*a
			});

		for (node_num, bias) in (0..32)
			.zip(self.layer1_bias_weights.iter_mut())
		{
			let conv_ind = node_num * 9;
			let out_ind = node_num * 26 * 26;
			let out = &d_l_d_i[
				out_ind..(out_ind + (26 * 26))
			];

			let mut conv_buf = vec![0.0; 9];

			do_convolute28_backprop(
				&self.input_image, out, &mut conv_buf
			);

			// Update the conv
			self.layer1_conv_weights[conv_ind..(conv_ind+9)].iter_mut()
				.zip(conv_buf.iter())
				.for_each(|(a, &b)| *a -= learning_rate * b);

			// Update the bias
			*bias -= learning_rate * out.iter().sum::<f64>();
		}
	}

	fn compute_layer1_maxpool(&mut self)
	{
		// Perform maxpooling
		for node_num in 0..32
		{
			let input_ind = node_num * 676;
			let pool_ind = node_num * 169;
			let input = &self.layer1_conv_bias_relu[
				input_ind..input_ind+676
			];
			let pool = &mut self.layer1_pooled[
				pool_ind..pool_ind+169
			];

			// Do the pooling
			do_pool26(input, pool);
		}
	}

	fn train_layer1_maxpool(&mut self, d_l_d_i: &[f64]) -> Vec<f64>
	{
		assert_eq!(d_l_d_i.len(), 13 * 13 * 32);
		let mut d_l_d_t = vec![0.0; 26 * 26 * 32];

		// populate d_l_d_t
		for node_num in 0..32 {
			let ind = node_num * 169;
			let input_ind = node_num * 676;

			let l = &mut d_l_d_t[input_ind..input_ind+676];

			let input = &self.layer1_conv_bias_relu[
				input_ind..input_ind+676
			];

			let out = &self.layer1_pooled[ind..ind+169];
			let loss = &d_l_d_i[ind..ind+169];

			do_pool26_backwards(input, out, l, loss);
		}

		d_l_d_t
	}

	fn compute_layer2_conv(&mut self)
	{
		for (node_num, &bias) in (0..64)
			.zip(self.layer2_bias_weights.iter())
		{
			let conv_ind = node_num * 9;
			let output_ind = node_num * 121 * 32;

			let conv = &self.layer2_conv_weights[conv_ind..conv_ind+9];
			let output = &mut self.layer2_conv[
				output_ind..output_ind + (121 * 32)
			];
			let output_bias_relu = &mut self.layer2_conv_bias_relu[
				output_ind..output_ind + (121 * 32)
			];

			for input_num in 0..32 {
				let out_ind = input_num * 121;
				let input_ind = input_num * 169;

				let out = &mut output[out_ind..out_ind+121];
				let input = &self.layer1_pooled[
					input_ind..input_ind+169
				];

				do_convolute13(input, out, conv);

				let out_bias_relu = &mut output_bias_relu[
					out_ind..out_ind+121
				];

				// Compute with bias + leaky_relu
				out.iter()
					.zip(out_bias_relu.iter_mut())
					.for_each(|(&a, b)| *b = leaky_relu(a + bias));
			}
		}
	}

	fn train_layer2_conv(&mut self,
		                 d_l_d_o: &[f64],
		                 learning_rate: f64) -> Vec<f64>
	{
		assert_eq!(d_l_d_o.len(), 121 * 32 * 64);

		for (node_num, bias) in (0..64)
			.zip(self.layer2_bias_weights.iter_mut())
		{
			let conv_ind = node_num * 9;
			let output_ind = node_num * 121 * 32;

			let loss = &d_l_d_o[
				output_ind..(output_ind + (121 * 32))
			];

			let mut conv_buf = [0.0; 9];
			for input_num in 0..32 {
				let out_ind = input_num * 121;
				let input_ind = input_num * 169;

				let mut out = loss[out_ind..out_ind+121].to_vec();
				// Step back through the relu
				out.iter_mut().for_each(|a| *a = if *a < 0.0 {
					0.01 * *a
				} else {
					*a
				});

				let input = &self.layer1_pooled[
					input_ind..input_ind+169
				];

				let mut cbuf = [0.0; 9];
				do_convolute13_backprop(input, &out, &mut cbuf);
				conv_buf.iter_mut()
					.zip(cbuf.iter())
					.for_each(|(a, &b)| *a += b);
			}

			// Update the conv
			self.layer2_conv_weights[conv_ind..conv_ind+9].iter_mut()
				.zip(conv_buf.iter())
				.for_each(|(a, &b)| *a -= learning_rate * b);

			// Update the bias
			// The bias is independent of the input and the conv
			*bias -= learning_rate * loss.iter().sum::<f64>();
		}

		let mut d_l_d_i = vec![0.0; 13 * 13 * 32];

		for input_num in 0..32 {
			let input_ind = input_num * (13 * 13);

			// The loss for a given node further back
			let d_l = &mut d_l_d_i[
				input_ind..(input_ind + (13 * 13))
			];

			// Each of our 64 nodes contributes
			for node_num in 0..64 {
				// Get out
				let output_ind = node_num * 121 * 32;
				let loss = &d_l_d_o[
					output_ind..(output_ind + (121 * 32))
				];
				let out_ind = input_num * 121;
				let out = &loss[out_ind..out_ind+121];
				let conv_ind = node_num * 9;
				let conv = &self.layer2_conv_weights[conv_ind..conv_ind+9];

				let mut u_conv_buf = vec![0.0; 13 * 13];

				do_convolute13_backprop_input(
					conv, out, &mut u_conv_buf,
				);

				// Update our d_l
				d_l.iter_mut()
					.zip(u_conv_buf.iter())
					.for_each(|(a, &b)| *a += b);
			}
		}

		d_l_d_i
	}

	fn compute_layer2_maxpool(&mut self)
	{
		for node_num in 0..64 {
			let conv_ind = node_num * 121 * 32;
			let convd = &mut self.layer2_conv_bias_relu[
				conv_ind..(conv_ind + (121 * 32))
			];

			let pool_ind = node_num * 25 * 32;
			let pooled = &mut self.layer2_pooled[
				pool_ind..(pool_ind + (25 * 32))
			];

			for input_num in 0..32 {
				let p_out_ind = input_num * 25;
				let i_ind = input_num * 121;

				let p_out = &mut pooled[p_out_ind..(p_out_ind+25)];
				let input = &convd[i_ind..(i_ind+121)];

				do_pool11(input, p_out);
			}
		}
	}

	fn train_layer2_maxpool(&mut self, d_l_d_o: &[f64]) -> Vec<f64>
	{
		// undo the maxpool
		assert_eq!(d_l_d_o.len(), 25 * 32 * 64);
		let mut d_l_d_i = vec![0.0; 121 * 32 * 64];

		for node_num in 0..64 {
			let conv_ind = node_num * 121 * 32;
			let convd = &self.layer2_conv_bias_relu[
				conv_ind..(conv_ind + (121 * 32))
			];

			let pool_ind = node_num * 25 * 32;
			let pooled = &self.layer2_pooled[
				pool_ind..(pool_ind + (25 * 32))
			];

			let loss = &d_l_d_o[
				pool_ind..(pool_ind + (25 * 32))
			];

			let back = &mut d_l_d_i[
				conv_ind..(conv_ind + (121 * 32))
			];

			for input_num in 0..32 {
				let p_out_ind = input_num * 25;
				let i_ind = input_num * 121;

				let p_out = &pooled[p_out_ind..(p_out_ind+25)];
				let input = &convd[i_ind..(i_ind+121)];

				let l = &loss[p_out_ind..(p_out_ind+25)];
				let b = &mut back[i_ind..(i_ind+121)];

				do_pool11_backprop(input, p_out, b, l);
			}
		}

		d_l_d_i
	}

	fn compute_layer3(&mut self)
	{
		// We have 64 nodes
		for (node_num, &bias) in (0..64)
			.zip(self.layer3_bias_weights.iter())
		{
			let node_ind = node_num * 5 * 5 * 32 * 64;
			let weights = &self.layer3_weights[
				node_ind..(node_ind + (5 * 5 * 32 * 64))
			];

			let sum = weights.iter()
				.zip(self.layer2_pooled.iter())
				.map(|(&a, &b)| a * b)
				.sum::<f64>();

			self.layer3_out[node_num] = leaky_relu(sum + bias);
		}
	}

	fn train_layer3(&mut self,
		            d_l_d_o: &[f64],
		            learning_rate: f64) -> Vec<f64>
	{
		assert_eq!(d_l_d_o.len(), 64);

		for ((node_num, &loss), bias) in (0..64)
			.zip(d_l_d_o.iter())
			.zip(self.layer3_bias_weights.iter_mut())
		{
			// Step back through the leaky_relu
			let loss = if loss < 0.0 {
				loss * 0.01
			} else {
				loss
			};

			// Update the bias
			*bias -= learning_rate * loss;

			// We have 51200 weights
			// These need to be evaluated at the input
			let weight_ind = node_num * 51200;
			let weights = &mut self.layer3_weights[
				weight_ind..(weight_ind + 51200)
			];

			for (wei, &inp) in weights.iter_mut()
				.zip(self.layer2_pooled.iter())
			{
				*wei -= learning_rate * loss * inp;
			}
		}

		// Compute d_l_d_i
		let mut d_l_d_i = vec![0.0; 5 * 5 * 64 * 32];
		for (n, i) in d_l_d_i.iter_mut().enumerate()
		{
			for (node_num, &loss) in (0..64)
				.zip(d_l_d_o.iter())
			{
				*i += loss * self.layer3_weights[
					(node_num * 51200) + n
				];
			}
		}

		d_l_d_i
	}

	fn compute_output_layer(&mut self, out: &mut Output)
	{
		// We have ten nodes - each has 64 weights
		for (node_num, &bias) in (0..10)
			.zip(self.output_bias_weights.iter())
		{
			// Sum over the weights
			let node_ind = node_num * 64;
			let weights = &self.output_weights[
				node_ind..node_ind+64
			];

			let ans = weights.iter()
				.zip(self.layer3_out.iter())
				.map(|(&a, &b)| a * b)
				.sum::<f64>();

			self.output_layer[node_num] = ans;
			self.output_layer_bias_exp[node_num] = 
				((ans + bias) * -1.0).exp();
		}

		// Apply the softmax
		let exp_sum = self.output_layer_bias_exp.iter().sum::<f64>();
		self.output_layer_bias_exp.iter()
			.zip(out.iter_mut())
			.for_each(|(&a, b)| *b = a / exp_sum);
	}

	fn train_output_layer(&mut self,
		                  d_loss: f64,
		                  label: usize,
		                  learning_rate: f64)
		-> Vec<f64>
	{
		let sum_exp = self.output_layer_bias_exp.iter().sum::<f64>();
		let sum_exp2 = sum_exp.powi(2);

		let mut d_l_d_t = [d_loss; 10];
		let exp_ans = self.output_layer_bias_exp[label];

		// Step back through the softmax
		for ((n, t), &e) in d_l_d_t.iter_mut().enumerate()
			.zip(self.output_layer_bias_exp.iter())
		{
			if n == label {
				*t *= (exp_ans * (sum_exp - exp_ans)) / sum_exp2;
			} else {
				*t *= (-1.0 * exp_ans * e) / sum_exp2;
			}
		}

		// Update the bias
		for (b, &l) in self.output_bias_weights.iter_mut()
			.zip(d_l_d_t.iter())
		{
			*b -= l * learning_rate;
		}

		// Update the weights
		for (node_num, &loss) in (0..10)
			.zip(d_l_d_t.iter())
		{
			let weights_ind = node_num * 64;
			let weights = &mut self.output_weights[
				weights_ind..(weights_ind + 64)
			];

			for (wei, &inp) in weights.iter_mut()
				.zip(self.layer3_out.iter())
			{
				*wei -= inp * loss * learning_rate;
			}
		}

		// Compute d_l_d_i
		let mut d_l_d_i = vec![0.0; 64];
		for (n, i) in d_l_d_i.iter_mut().enumerate()
		{
			// Each input affects all ten nodes.
			for (node_num, &loss) in (0..10)
				.zip(d_l_d_t.iter())
			{
				let w = self.output_weights[
					(node_num * 64) + n
				];
				*i += w * loss;
			}
		}

		d_l_d_i
	}

}

fn dist() -> f64 {
	let a = StdRng::from_entropy().sample(Uniform::from(-500..500));
	a as f64 / 1000.0
}

fn do_convolute28(input: &[f64], out: &mut [f64], conv: &[f64])
{
	assert_eq!(input.len(), 28 * 28);
	assert_eq!(out.len(), 26 * 26);
	assert_eq!(conv.len(), 9);

	unsafe {
		convolute28(input.as_ptr(), out.as_mut_ptr(), conv.as_ptr());
	}
}

fn do_convolute28_backprop(input: &[f64],
	                       output: &[f64],
	                       d_conv: &mut [f64])
{
	assert_eq!(28 * 28, input.len());
	assert_eq!(26 * 26, output.len());
	assert_eq!(9, d_conv.len());

	unsafe {
		convoluted28_backprop(input.as_ptr(), output.as_ptr(),
			                  d_conv.as_mut_ptr());
	}
}

fn do_pool26(input: &[f64], out: &mut [f64])
{
	assert_eq!(input.len(), 26 * 26);
	assert_eq!(out.len(), 13 * 13);

	unsafe {
		pool26(input.as_ptr(), out.as_mut_ptr());
	}
}

fn do_pool26_backwards(input: &[f64],
	                   output: &[f64],
	                   back: &mut [f64],
	                   d_loss: &[f64])
{
	assert_eq!(26 * 26, input.len());
	assert_eq!(13 * 13, output.len());
	assert_eq!(26 * 26, back.len());
	assert_eq!(13 * 13, d_loss.len());

	unsafe {
		let e = pool26_backprop(input.as_ptr(),
			                    output.as_ptr(),
			                    back.as_mut_ptr(),
			                    d_loss.as_ptr());
		if e != 0 {
			panic!("pool26_backprop returned error");
		}
	}
}

fn do_convolute13(input: &[f64], out: &mut [f64], conv: &[f64])
{
	assert_eq!(input.len(), 13 * 13);
	assert_eq!(out.len(), 11 * 11);
	assert_eq!(conv.len(), 9);

	unsafe {
		convolute13(input.as_ptr(), out.as_mut_ptr(), conv.as_ptr());
	}
}

fn do_convolute13_backprop(input: &[f64], output: &[f64],
	                       d_conv: &mut [f64]) {
	assert_eq!(13 * 13, input.len());
	assert_eq!(11 * 11, output.len());
	assert_eq!(9, d_conv.len());

	unsafe {
		convoluted13_backprop(input.as_ptr(), output.as_ptr(),
			                  d_conv.as_mut_ptr());
	}
}

fn do_convolute13_backprop_input(input: &[f64],
						         output: &[f64],
						         d_conv: &mut [f64])
{
	assert_eq!(input.len(), 9);
	assert_eq!(output.len(), 11 * 11);
	assert_eq!(d_conv.len(), 13 * 13);

	unsafe {
		convoluted13_backprop_input(
			input.as_ptr(),
			output.as_ptr(),
			d_conv.as_mut_ptr(),
		);
	}
}

fn do_pool11(input: &[f64], output: &mut [f64]) {
	assert_eq!(11 * 11, input.len());
	assert_eq!(5 * 5, output.len());

	unsafe {
		pool11(input.as_ptr(), output.as_mut_ptr());
	}
}

fn do_pool11_backprop(input: &[f64],
	                  output: &[f64],
	                  back: &mut [f64],
	                  d_loss: &[f64]) {

	assert_eq!(11 * 11, input.len());
	assert_eq!(5 * 5, output.len());
	assert_eq!(11 * 11, back.len());
	assert_eq!(5 * 5, d_loss.len());

	unsafe {
		let e = pool11_backprop(input.as_ptr(),
			                    output.as_ptr(),
			                    back.as_mut_ptr(),
			                    d_loss.as_ptr());
		if e != 0 {
			panic!("pool11_backprop returned error");
		}
	}
}

fn leaky_relu(v: f64) -> f64 {
	if v < 0.0 {
		0.01 * v
	} else {
		v
	}
}