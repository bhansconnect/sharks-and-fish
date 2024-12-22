use serde::Deserialize;

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Config {
    pub sim: Sim,
    pub target_fps: u32,
    pub sharks: Sharks,
    pub fish: Fish,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Sim {
    pub width: f32,
    pub height: f32,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct Sharks {
    // pub count: u32,
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
}
