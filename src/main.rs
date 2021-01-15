use std::fmt;
use std::fs;
use std::io::{BufReader, Read};

use rand::thread_rng;
use rand::seq::SliceRandom;

mod net;
mod errors;
use errors::*;


const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";


#[derive(Clone, Default)]
struct Image {
	pixels: Box<[[f64; 28]; 28]>,
	label: u8,
}

impl fmt::Display for Image {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for row in self.pixels.iter() {
			for &col in row.iter() {
				write!(f, "{}", if col < 0.25 {
						' '
					} else if col < 0.5{
						'_'
					} else if col < 0.75 {
						'-'
					} else {
						'='
					}
				)?;
			}
			writeln!(f)?;
		}
		write!(f, "label = {}", self.label)
	}
}

fn open_file(name: &str) -> Result<BufReader<fs::File>> {

	Ok(BufReader::new(
		fs::OpenOptions::new()
			.read(true)
			.open(name)?
		))
}

fn get_data() -> Result<(Vec<Image>, Vec<Image>)> {
	let mut train_images = open_file(TRAIN_IMAGES)?;
	let mut train_labels = open_file(TRAIN_LABELS)?;
	let mut test_images = open_file(TEST_IMAGES)?;
	let mut test_labels = open_file(TEST_LABELS)?;

	// Buffer
	let mut buf = [0 as u8; 4];

	// Drain the header of the images
	train_images.read_exact(&mut buf)?;
	assert_eq!(2051, u32::from_be_bytes(buf));
	train_images.read_exact(&mut buf)?;
	assert_eq!(60000, u32::from_be_bytes(buf));
	train_images.read_exact(&mut buf)?;
	assert_eq!(28, u32::from_be_bytes(buf));
	train_images.read_exact(&mut buf)?;
	assert_eq!(28, u32::from_be_bytes(buf));

	// Drain the header of the labels
	train_labels.read_exact(&mut buf)?;
	assert_eq!(2049, u32::from_be_bytes(buf));
	train_labels.read_exact(&mut buf)?;
	assert_eq!(60000, u32::from_be_bytes(buf));

	// Drain the header of the images
	test_images.read_exact(&mut buf)?;
	assert_eq!(2051, u32::from_be_bytes(buf));
	test_images.read_exact(&mut buf)?;
	assert_eq!(10000, u32::from_be_bytes(buf));
	test_images.read_exact(&mut buf)?;
	assert_eq!(28, u32::from_be_bytes(buf));
	test_images.read_exact(&mut buf)?;
	assert_eq!(28, u32::from_be_bytes(buf));

	// Drain the header of the labels
	test_labels.read_exact(&mut buf)?;
	assert_eq!(2049, u32::from_be_bytes(buf));
	test_labels.read_exact(&mut buf)?;
	assert_eq!(10000, u32::from_be_bytes(buf));

	// Load the data
	let mut train = Vec::with_capacity(60000);
	let mut test = Vec::with_capacity(10000);
	let mut pixel_buf = vec![0; 28 * 28];

	for l in train_labels.bytes() {
		train_images.read_exact(&mut *pixel_buf)?;

		// Copy the pixel_buf
		let mut pixels = Box::new([[0.0; 28]; 28]);
		for (p1, &p2) in pixels.iter_mut().flatten()
			.zip(pixel_buf.iter()) {

			*p1 = p2 as f64 / u8::MAX as f64;
		}

		train.push(Image{
			label: l?,
			pixels: pixels,
		});
	}

	assert_eq!(60000, train.len());

	for l in test_labels.bytes() {
		test_images.read_exact(&mut *pixel_buf)?;

		let mut pixels = Box::new([[0.0; 28]; 28]);
		for (p1, &p2) in pixels.iter_mut().flatten()
			.zip(pixel_buf.iter()) {

			*p1 = p2 as f64 / u8::MAX as f64;
		}

		test.push(Image{
			label: l?,
			pixels: pixels,
		});
	}
	assert_eq!(10000, test.len());

	Ok((train, test))
}

fn compute_total_loss(net: &mut net::Net, test_images: &[Image]) -> Result<f64> {
	let mut total_loss = 0.0;
	let mut answer = [0.0; 10];
	let mut below_95 = 0;
	let mut below_95_arr = [0; 10];
	let mut above_99 = 0;
	let mut above_99_arr = [0; 10];

	for i in test_images.iter() {
		net.compute(&i.pixels, &mut answer);
		let loss = -1.0 * answer[i.label as usize].ln();
		total_loss += loss;
		if answer[i.label as usize] < 0.95 {
			below_95 += 1;
			below_95_arr[i.label as usize] += 1;
		} else if answer[i.label as usize] > 0.99 {
			above_99 += 1;
			above_99_arr[i.label as usize] += 1;
		}
	}

	println!("total_loss = {}", total_loss);
	println!("total below_95 = {}", below_95);
	println!("below_95_arr = {:?}", below_95_arr);
	println!("total_above_99 = {}", above_99);
	println!("above_99_arr = {:?}", above_99_arr);

	Ok(total_loss)
}


fn run() -> Result<()> {
	// Read the Mnist files
	let (mut train_images, mut test_images) = get_data()?;

	// Create the NET
	let mut net = net::Net::new();
	println!("starting");

	let mut answer = [0.0; 10];
	let mut learning_rate = 0.0005;

	for epoch in 0..32
	{
		train_images.shuffle(&mut thread_rng());

		for i in train_images[0..512].iter() {
			net.compute(&i.pixels, &mut answer);
			for a in answer.iter() {
				if a.is_nan() || a.is_infinite() {
					println!("net has collapsed");
					return Ok(());
				}
			}
			net.train(i.label as usize, &answer, learning_rate);
		}

		print!("done epoch {}\t", epoch);
		compute_total_loss(&mut net, &test_images[0..512])?;
	}

	// Run four times over the same set of images
	learning_rate /= 10.0;
	for epoch in 0..4
	{
		train_images.shuffle(&mut thread_rng());

		for i in train_images.iter() {
			net.compute(&i.pixels, &mut answer);
			for a in answer.iter() {
				if a.is_nan() || a.is_infinite() {
					println!("net has collapsed");
					return Ok(());
				}
			}
			net.train(i.label as usize, &answer, learning_rate);
		}

		print!("done epoch {}\t", epoch);
		compute_total_loss(&mut net, &test_images[0..512])?;
		println!();

		if epoch == 1 {
			learning_rate /= 10.0;
		}
	}

	let mut total_loss = 0.0;
	let mut below_95 = 0;
	let mut below_95_arr = [0; 10];
	let mut above_99 = 0;
	let mut above_99_arr = [0; 10];

	test_images.shuffle(&mut thread_rng());

	for i in test_images[0..1024].iter() {
		net.compute(&i.pixels, &mut answer);
		println!("{}", i);
		println!("{:?}", answer);
		println!("res = {}", answer[i.label as usize]);
		let loss = -1.0 * answer[i.label as usize].ln();
		println!("loss = {}", loss);
		total_loss += loss;

		if answer[i.label as usize] < 0.95 {
			below_95 += 1;
			below_95_arr[i.label as usize] += 1;
		} else if answer[i.label as usize] > 0.99 {
			above_99 += 1;
			above_99_arr[i.label as usize] += 1;
		}
	}

	println!("total_loss = {}", total_loss);
	println!("total below_95 = {}", below_95);
	println!("below_95_arr = {:?}", below_95_arr);
	println!("total_above_99 = {}", above_99);
	println!("above_99_arr = {:?}", above_99_arr);

	Ok(())
}


fn main() {
	if let Err(e) = run() {
		eprintln!("something went wrong {}", e.to_string());
	}
}
