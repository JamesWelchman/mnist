
fn main() {
	cc::Build::new()
		.file("src/operations.c")
		.flag("--std=c99")
		.compile("operations");
}