use crossbeam::atomic::AtomicCell;
use std::{
    fmt,
    sync::Arc,
    time::{Instant}, path::PathBuf, io::{Read, Write}, fs::File,
};
use clap::{Parser, clap_derive::ArgEnum};

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[clap(short, long, value_parser, value_name = "FILE")]
    file: Option<PathBuf>,

    #[clap(short, long)]
    n: Option<usize>,

    #[clap(short, long)]
    m: Option<usize>,

    #[clap(short, long)]
    k: Option<usize>,

    #[clap(short, long, arg_enum, value_parser)]
    mode: Mode
}

#[derive(Clone, Copy, PartialEq, Eq, ArgEnum, Debug)]
enum Mode {
    Seq,
    Par,
    All
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let matrix1;
    let matrix2;

    if let Some(file_path) = args.file {
        let mut data = String::new();
        let mut file = File::open(file_path).expect("Unable to open file");
        file.read_to_string(&mut data).expect("Unable to read string");
        let splitted: Vec<String> = data.split("X").map(|x| x.trim().to_owned()).collect();
        matrix1 = Matrix::from_string(&splitted[0]);
        matrix2 = Matrix::from_string(&splitted[1]);
    } else {
        let n = args.n.expect("No n found");
        let m = args.m.expect("No m found");
        let k = args.k.expect("No k found");
        matrix1 = Matrix::random(n, m);
        matrix2 = Matrix::random(m, k);
    }

    if args.mode == Mode::Seq {
        let start = Instant::now();
        let matrix_res = matrix1.multiply(&matrix2);
        let elapsed = start.elapsed();
        println!("Done! Elapsed time: {:?}", elapsed);
        matrix_res.write_to_file();
    } else if args.mode == Mode::Par {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let start = Instant::now();
        let matrix_res = pool.install(|| matrix1.multiply_par(&matrix2));
        let elapsed = start.elapsed();
        println!("Done! Elapsed time: {:?}", elapsed);
        matrix_res.write_to_file();
    } else {
        let start = Instant::now();
        let matrix_res_seq = matrix1.multiply(&matrix2);
        let elapsed = start.elapsed();
        println!("Done! Elapsed time for SEQ: {:?}", elapsed);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let start = Instant::now();
        let matrix_res_par = pool.install(|| matrix1.multiply_par(&matrix2));
        let elapsed = start.elapsed();
        println!("Done! Elapsed time for PAR: {:?}", elapsed);

        assert_eq!(matrix_res_seq, matrix_res_par);

        matrix_res_par.write_to_file();
    }
}

#[derive(Clone, Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{} ", self.data[i * self.cols + j])?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Matrix {
    fn new(rows: usize, cols: usize, vec: Vec<f64>) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec,
        }
    }

    fn from_string(s: &String) -> Matrix {
        let mut rows = 0;
        let mut cols = 0;
        let mut data = Vec::new();

        for line in s.lines() {
            let splitted: Vec<String> = line.split(" ").map(|x| x.to_owned()).collect();

            if splitted.len() == 0 {
                continue;
            }

            if cols == 0 {
                cols = splitted.len();
            } else if cols != splitted.len() {
                panic!("Cannot read matrix");
            }

            rows += 1;

            for num_str in splitted {
                let num = num_str.parse::<f64>().expect("Not a number");
                data.push(num);
            }
        }

        Matrix { rows, cols, data }
    }

    fn random(rows: usize, cols: usize) -> Matrix {
        let mut m = Matrix::new(rows, cols, vec![0.0; rows * cols]);
        for i in 0..m.data.len() {
            m.data[i] = rand::random::<f64>();
        }
        m
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        let mut result = Matrix::new(self.rows, other.cols, vec![0.0; self.rows * other.cols]);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }

                result.set(i, j, sum);
            }
        }

        result
    }

    fn multiply_par(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        let result = Arc::new(AtomicCell::new(Matrix::new(
            self.rows,
            other.cols,
            vec![0.0; self.rows * other.cols],
        )));

        rayon::scope(|s| {
            for i in 0..self.rows {
                for j in 0..other.cols {
                    let result = Arc::clone(&result);
                    s.spawn(move |_| {
                        let mut sum = 0.0;

                        for k in 0..self.cols {
                            sum += self.get(i, k) * other.get(k, j);
                        }

                        unsafe {
                            (*result.as_ptr()).set(i, j, sum);
                        }
                    });
                }
            }
        });

        unsafe { (*result.as_ptr()).clone() }
    }

    fn write_to_file(&self) {
        let mut file = File::create("output.txt").expect("Unable to create file");
        file.write_all(format!("{}", self).as_bytes()).expect("Unable to write data");
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for i in 0..self.data.len() {
            if self.data[i] != other.data[i] {
                return false;
            }
        }

        true
    }
}

#[macro_export]
macro_rules! matrix {
    () => {
        {
            Matrix::new(0, 0, vec![])
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            let rows = data_as_nested_array.len();
            let cols = data_as_nested_array[0].len();
            let data_as_flat_array: Vec<f64> = data_as_nested_array.into_iter()
                .flat_map(|row| row.into_iter())
                .collect();
            Matrix::new(rows, cols, data_as_flat_array)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_identity() {
        let a = matrix![
            1.0, 0.0;
            0.0, 1.0
        ];
        let b = matrix![
            1.0, 4.0;
            2.0, 3.0
        ];
        let expected = matrix![
            1.0, 4.0;
            2.0, 3.0
        ]; 

        assert_eq!(a.multiply(&b), expected);
    }

    #[test]
    fn mul_identity_par() {
        let a = matrix![
            1.0, 0.0;
            0.0, 1.0
        ];
        let b = matrix![
            1.0, 4.0;
            2.0, 3.0
        ];
        let expected = matrix![
            1.0, 4.0;
            2.0, 3.0
        ]; 

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let c = pool.install(|| a.multiply_par(&b));

        assert_eq!(c, expected);
    }

    #[test]
    fn mul_not_squared() {
        let a = matrix![1.0, 0.0];
        let b = matrix![1.0;
                                2.0];
        let expected = matrix![1.0]; 

        assert_eq!(a.multiply(&b), expected);
    }

    #[test]
    fn mul_not_squared_par() {
        let a = matrix![1.0, 0.0];
        let b = matrix![1.0;
                                2.0];
        let expected = matrix![1.0]; 

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let c = pool.install(|| a.multiply_par(&b));

        assert_eq!(c, expected);
    }

    #[test]
    fn mul_squared() {
        let a = matrix![1.0, 2.0;
                                3.0, 4.0];
        let b = matrix![1.0, 2.0;
                                3.0, 4.0];
        let expected = matrix![7.0, 10.0;
                                       15.0, 22.0]; 

        assert_eq!(a.multiply(&b), expected);
    }

    #[test]
    fn mul_squared_par() {
        let a = matrix![1.0, 2.0;
                                3.0, 4.0];
        let b = matrix![1.0, 2.0;
                                3.0, 4.0];
        let expected = matrix![7.0, 10.0;
                                       15.0, 22.0];

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let c = pool.install(|| a.multiply_par(&b));

        assert_eq!(c, expected);
    }
}