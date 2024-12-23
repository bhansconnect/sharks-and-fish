use burn::prelude::*;
use rand::Rng;
use rand_pcg::Pcg64;

// All the AI...
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear0: nn::Linear<B>,
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    activation: nn::Relu,
}

impl<B: Backend> Model<B> {
    fn consume(self) -> (nn::Linear<B>, nn::Linear<B>, nn::Linear<B>) {
        (self.linear0, self.linear1, self.linear2)
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.linear0.forward(x));
        let x = self.activation.forward(self.linear1.forward(x));
        self.activation.forward(self.linear2.forward(x))
    }

    pub fn pick_action(&self, x: Tensor<B, 1>, step: u64, rng: &mut Pcg64) -> u32 {
        if rng.gen::<f32>() > eps_threshold(step) {
            self.forward(x.unsqueeze())
                .argmax(1)
                .to_data()
                .as_slice::<i32>()
                .unwrap()[0] as u32
        } else {
            let output_size = self.linear2.weight.shape().dims[1];
            rng.gen_range(0..output_size) as u32
        }
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear0: nn::LinearConfig::new(self.input_size, self.hidden_size).init(device),
            linear1: nn::LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear2: nn::LinearConfig::new(self.hidden_size, self.output_size).init(device),
            activation: nn::Relu::new(),
        }
    }
}

const EPS_DECAY: f32 = 10000.0;
const EPS_START: f32 = 0.9;
const EPS_END: f32 = 0.05;

fn eps_threshold(step: u64) -> f32 {
    EPS_END + (EPS_START - EPS_END) * f32::exp(-(step as f32) / EPS_DECAY)
}
