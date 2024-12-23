use burn::{
    grad_clipping::GradientClippingConfig,
    module::{Param, ParamId},
    optim::{adaptor::OptimizerAdaptor, AdamW, AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::{MseLoss, Reduction};
use rand::Rng;
use rand_pcg::Pcg64;
use ringbuffer::{GrowableAllocRingBuffer, RingBuffer};

pub struct Hist<B: Backend> {
    pub state: Tensor<B, 1>,
    pub action: u32,
    pub reward: f32,
    pub next_state: Tensor<B, 1>,
}

pub struct DQNTrainer<'a, B: AutodiffBackend> {
    pub network: Model<B>,
    target: Model<B>,
    pub memory: GrowableAllocRingBuffer<Hist<B>>,
    batch_size: usize,
    device: &'a Device<B>,
    optimizer: OptimizerAdaptor<AdamW<B::InnerBackend>, Model<B>, B>,
    tau: f32,
    gamma: f32,
    learning_rate: f32,
    eps_decay: f32,
}

const EPS_START: f32 = 0.9;
const EPS_END: f32 = 0.05;

impl<'a, B: AutodiffBackend> DQNTrainer<'a, B> {
    pub fn push(&mut self, hist: Hist<B>) {
        self.memory.push(hist);
    }

    pub fn train(&mut self, rng: &mut Pcg64) {
        if self.memory.len() < self.batch_size {
            return;
        }

        let (states, next_states, actions, rewards) = self.get_batch(rng);
        let actions = actions.reshape([self.batch_size as i32, 1]);
        let state_action_values = self.network.forward(states).gather(1, actions);
        let next_state_values = self.target.forward(next_states).max_dim(1).detach();

        let rewards = rewards.reshape([self.batch_size as i32, 1]);
        let expected_state_action_values = next_state_values.mul_scalar(self.gamma) + rewards;

        let loss = MseLoss.forward(
            state_action_values,
            expected_state_action_values,
            Reduction::Mean,
        );

        let gradients = loss.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &self.network);
        let mut dummy = Model {
            linear0: nn::LinearConfig::new(0, 0).init(self.device),
            linear1: nn::LinearConfig::new(0, 0).init(self.device),
            linear2: nn::LinearConfig::new(0, 0).init(self.device),
            activation: nn::Relu::new(),
        };
        std::mem::swap(&mut dummy, &mut self.network);
        self.network = self
            .optimizer
            .step(self.learning_rate as f64, dummy, gradient_params);
        // Shift target slowly.
        self.target.linear0.weight = soft_update_tensor(
            &self.target.linear0.weight,
            &self.network.linear0.weight,
            self.tau,
        );
        self.target.linear0.bias = Some(soft_update_tensor(
            self.target.linear0.bias.as_ref().unwrap(),
            self.network.linear0.bias.as_ref().unwrap(),
            self.tau,
        ));
        self.target.linear1.weight = soft_update_tensor(
            &self.target.linear1.weight,
            &self.network.linear1.weight,
            self.tau,
        );
        self.target.linear1.bias = Some(soft_update_tensor(
            self.target.linear1.bias.as_ref().unwrap(),
            self.network.linear1.bias.as_ref().unwrap(),
            self.tau,
        ));
        self.target.linear2.weight = soft_update_tensor(
            &self.target.linear2.weight,
            &self.network.linear2.weight,
            self.tau,
        );
        self.target.linear2.bias = Some(soft_update_tensor(
            self.target.linear2.bias.as_ref().unwrap(),
            self.network.linear2.bias.as_ref().unwrap(),
            self.tau,
        ));
    }

    fn sample_indices(&self, rng: &mut Pcg64) -> Vec<usize> {
        let mut sample = Vec::<usize>::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            let index = rng.gen_range(0..self.memory.len());
            sample.push(index);
        }

        sample
    }

    fn get_batch(
        &self,
        rng: &mut Pcg64,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1, Int>, Tensor<B, 1>) {
        let indices = self.sample_indices(rng);

        let mut states = Vec::with_capacity(self.batch_size);
        let mut next_states = Vec::with_capacity(self.batch_size);
        let mut actions = Vec::with_capacity(self.batch_size);
        let mut rewards = Vec::with_capacity(self.batch_size);
        for hist in indices.into_iter().filter_map(|i| self.memory.get(i)) {
            states.push(hist.state.clone().unsqueeze());
            next_states.push(hist.next_state.clone().unsqueeze());
            actions.push(hist.action);
            rewards.push(hist.reward);
        }
        let states = Tensor::cat(states, 0);
        let next_states = Tensor::cat(next_states, 0);
        let actions = Tensor::from_ints(actions.as_slice(), self.device);
        let rewards = Tensor::from_floats(rewards.as_slice(), self.device);
        (states, next_states, actions, rewards)
    }

    pub fn eps_threshold(&self, step: u64) -> f32 {
        EPS_END + (EPS_START - EPS_END) * f32::exp(-(step as f32) / self.eps_decay)
    }

    pub fn pick_action(&self, x: Tensor<B, 1>, step: u64, rng: &mut Pcg64) -> u32 {
        if rng.gen::<f32>() > self.eps_threshold(step) {
            self.network
                .forward(x.unsqueeze())
                .argmax(1)
                .to_data()
                .as_slice::<i32>()
                .unwrap()[0] as u32
        } else {
            let output_size = self.network.linear2.weight.shape().dims[1];
            rng.gen_range(0..output_size) as u32
        }
    }
}

// All the AI...
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear0: nn::Linear<B>,
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    activation: nn::Relu,
}

impl<B: Backend> Model<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.linear0.forward(x));
        let x = self.activation.forward(self.linear1.forward(x));
        self.activation.forward(self.linear2.forward(x))
    }
}

fn soft_update_tensor<const N: usize, B: Backend>(
    this: &Param<Tensor<B, N>>,
    that: &Param<Tensor<B, N>>,
    tau: f32,
) -> Param<Tensor<B, N>> {
    let that_weight = that.val();
    let this_weight = this.val();
    let new_weight = this_weight * (1.0 - tau) + that_weight * tau;

    Param::initialized(ParamId::new(), new_weight)
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    replay_buffer_size: usize,
    batch_size: usize,
    tau: f32,
    gamma: f32,
    learning_rate: f32,
    eps_decay: f32,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<'a, B: AutodiffBackend>(&self, device: &'a B::Device) -> DQNTrainer<'a, B> {
        let model = Model {
            linear0: nn::LinearConfig::new(self.input_size, self.hidden_size).init(device),
            linear1: nn::LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear2: nn::LinearConfig::new(self.hidden_size, self.output_size).init(device),
            activation: nn::Relu::new(),
        };
        let optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Value(100.0)))
            .init();
        DQNTrainer {
            network: model.clone(),
            target: model,
            memory: GrowableAllocRingBuffer::with_capacity(self.replay_buffer_size),
            batch_size: self.batch_size,
            device,
            optimizer,
            tau: self.tau,
            gamma: self.gamma,
            learning_rate: self.learning_rate,
            eps_decay: self.eps_decay,
        }
    }
}
