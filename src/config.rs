use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Config {
    pub sim: Sim,
    pub target_fps: u32,
    pub shark: Shark,
    pub fish: Fish,
    pub dqn: DQN,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Sim {
    pub width: f32,
    pub height: f32,
    pub sims_per_frame: u32,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Shark {
    pub vision_rays: u32,
    pub vision_dist: f32,
    pub vision_angle: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub max_force: f32,
    pub max_reverse_force: f32,
    pub max_torque: f32,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Fish {
    pub count: u32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub max_force: f32,
    pub max_reverse_force: f32,
    pub max_torque: f32,
    pub mutation_factor: f32,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct DQN {
    pub hidden_size: u32,
    pub replay_buffer_size: u32,
    pub train_freq: u32,
    pub batch_size: u32,
    pub tau: f32,
    pub gamma: f32,
    pub learning_rate: f32,
    pub eps_decay: f32,
    pub reset_delay: u32,
}
