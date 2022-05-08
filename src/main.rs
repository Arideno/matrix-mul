use crossbeam::atomic::AtomicCell;
use rand::Rng;
use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

const TIMES: usize = 1000;

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

fn main() {
    benchmark();
}

fn benchmark() {
    let mut rng = rand::thread_rng();

    let mut sync_time = [Duration::ZERO; TIMES];
    let mut async_time = [Duration::ZERO; TIMES];

    for t in 0..TIMES {
        let i: usize = rng.gen_range(500..=1000);
        let j: usize = rng.gen_range(500..=1000);
        let k: usize = rng.gen_range(500..=1000);
        let a = Matrix::random(i, k);
        let b = Matrix::random(k, j);

        println!("{t} - Generated");

        let start = Instant::now();

        let c = a.multiply(&b);

        sync_time[t] = start.elapsed();

        println!("Sync Elapsed: {:?}", sync_time[t]);

        let start = Instant::now();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let d = pool.install(|| a.multiply_async(&b));

        async_time[t] = start.elapsed();

        println!("Async Elapsed: {:?}", async_time[t]);

        assert_eq!(c, d);
    }

    println!(
        "Sync Average: {:?}",
        sync_time.iter().sum::<Duration>() / TIMES as u32
    );
    println!(
        "Async Average: {:?}",
        async_time.iter().sum::<Duration>() / TIMES as u32
    );
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

    fn multiply_async(&self, other: &Matrix) -> Matrix {
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
