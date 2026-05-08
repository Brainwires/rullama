use std::env; use std::fs;
use rullama::gguf::GgufReader;
fn main() {
    let path = env::args().nth(1).unwrap();
    let bytes = fs::read(&path).unwrap();
    let r = GgufReader::new(&bytes).unwrap();
    let mut names: Vec<_> = r.tensors().iter().map(|t| (t.name.clone(), format!("{:?}", t.dtype), format!("{:?}", t.dims))).collect();
    names.sort();
    for (n, d, dims) in &names {
        if n.starts_with("blk.0.") || n.starts_with("blk.1.") || !n.starts_with("blk.") {
            // print non-blk tensors, plus blk.0 and blk.1 for comparison
            if !n.starts_with("v.") && !n.starts_with("a.") {
                println!("{:>7} {:<20} {}", d, dims, n);
            }
        }
    }
}
