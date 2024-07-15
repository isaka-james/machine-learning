pub mod linreg {
    pub struct LinearRegression {
        pub slope: f64,
        pub intercept: f64,
        pub r_squared: f64,
        pub residuals: Vec<f64>,
    }

    impl LinearRegression {
        pub fn new() -> Self {
            LinearRegression {
                slope: 0.0,
                intercept: 0.0,
                r_squared: 0.0,
                residuals: Vec::new(),
            }
        }

        pub fn fit(&mut self, x: &[f64], y: &[f64]) -> Result<(), &'static str> {
            if x.len() != y.len() {
                return Err("The length of x and y must be the same.");
            }

            if x.is_empty() || y.is_empty() {
                return Err("Input arrays cannot be empty.");
            }

            let n = x.len() as f64;

            let sum_x: f64 = x.iter().sum();
            let sum_y: f64 = y.iter().sum();
            let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
            let sum_x_squared: f64 = x.iter().map(|xi| xi * xi).sum();

            self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
            self.intercept = (sum_y - self.slope * sum_x) / n;

            self.calculate_r_squared(x, y);
            self.calculate_residuals(x, y);
            Ok(())
        }

        pub fn predict(&self, x: f64) -> f64 {
            self.slope * x + self.intercept
        }

        fn calculate_r_squared(&mut self, x: &[f64], y: &[f64]) {
            let mean_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
            let total_variance: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
            let residual_variance: f64 = x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| (yi - self.predict(*xi)).powi(2))
                .sum();
            self.r_squared = 1.0 - (residual_variance / total_variance);
        }

        fn calculate_residuals(&mut self, x: &[f64], y: &[f64]) {
            self.residuals = x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| yi - &self.predict(*xi))
                .collect();
        }

        pub fn get_r_squared(&self) -> f64 {
            self.r_squared
        }

        pub fn get_residuals(&self) -> &[f64] {
            &self.residuals
        }

        pub fn mean_squared_error(&self) -> f64 {
            let mse: f64 = self.residuals.iter().map(|res| res.powi(2)).sum::<f64>() / self.residuals.len() as f64;
            mse
        }

        pub fn root_mean_squared_error(&self) -> f64 {
            self.mean_squared_error().sqrt()
        }
    }
}

pub mod logisticreg {
    pub struct LogisticRegression {
        coefficients: Vec<f64>,
    }

    impl LogisticRegression {
        pub fn new(num_features: usize) -> Self {
            let coefficients = vec![0.0; num_features + 1];
            LogisticRegression { coefficients }
        }

        fn sigmoid(&self, z: f64) -> f64 {
            1.0 / (1.0 + (-z).exp())
        }

        pub fn fit(&mut self, xlarge: &[Vec<f64>], y: &[f64], learning_rate: f64, epochs: usize) {
            let num_samples = xlarge.len();
            let num_features = xlarge[0].len();

            for _ in 0..epochs {
                for i in 0..num_samples {
                    let mut linear_model = self.coefficients[0];
                    for j in 0..num_features {
                        linear_model += self.coefficients[j + 1] * xlarge[i][j];
                    }
                    let prediction = self.sigmoid(linear_model);
                    let error = y[i] - prediction;

                    self.coefficients[0] += learning_rate * error;
                    for j in 0..num_features {
                        self.coefficients[j + 1] += learning_rate * error * xlarge[i][j];
                    }
                }
            }
        }

        pub fn predict_proba(&self, xlarge: &[Vec<f64>]) -> Vec<f64> {
            let mut probabilities = Vec::new();
            for x in xlarge.iter() {
                let mut linear_model = self.coefficients[0];
                for j in 0..x.len() {
                    linear_model += self.coefficients[j + 1] * x[j];
                }
                probabilities.push(self.sigmoid(linear_model));
            }
            probabilities
        }

        pub fn predict(&self, xlarge: &[Vec<f64>]) -> Vec<u8> {
            self.predict_proba(xlarge)
                .iter()
                .map(|&p| if p >= 0.5 { 1 } else { 0 })
                .collect()
        }
    }
}
