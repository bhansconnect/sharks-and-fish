use notify::{RecursiveMode::NonRecursive, Watcher};
use rand::prelude::*;
use rand_pcg::Pcg64;
use rapier2d::prelude::*;
use raylib::prelude::*;
use std::{
    mem::MaybeUninit,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

mod config;
mod dqn;
use config::Config;

const DEFAULT_WIDTH: i32 = 1280;
const DEFAULT_HEIGHT: i32 = 720;
const MAIN_GROUP: Group = Group::GROUP_1;
const EDIBLE_GROUP: Group = Group::GROUP_2;

fn main() {
    pretty_env_logger::init();
    init_config();
    let init_config = load_config();
    let (mut rl, thread) = raylib::init()
        .log_level(raylib::consts::TraceLogLevel::LOG_WARNING)
        .size(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        .resizable()
        .title("Sharks and Fish")
        .vsync()
        .build();

    // Use fps to set sim speed.
    // Can run multiple physic steps per frame to run faster.
    // Can also turn of visualization to run super fast.
    rl.set_target_fps(init_config.target_fps);

    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    // Create the walls.
    let thickness = 1.0;
    let mut walls = vec![];
    walls.push(
        collider_set.insert(
            ColliderBuilder::cuboid(init_config.sim.width / 2.0, thickness)
                .translation(vector![0.0, 0.5 * init_config.sim.height + thickness / 2.0])
                .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
                .restitution(1.0)
                .friction(0.0)
                .build(),
        ),
    );
    walls.push(
        collider_set.insert(
            ColliderBuilder::cuboid(init_config.sim.width / 2.0, thickness)
                .translation(vector![
                    0.0,
                    -0.5 * init_config.sim.height - thickness / 2.0
                ])
                .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
                .restitution(1.0)
                .friction(0.0)
                .build(),
        ),
    );
    walls.push(
        collider_set.insert(
            ColliderBuilder::cuboid(thickness, init_config.sim.height / 2.0)
                .translation(vector![0.5 * init_config.sim.width + thickness / 2.0, 0.0])
                .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
                .restitution(1.0)
                .friction(0.0)
                .build(),
        ),
    );
    walls.push(
        collider_set.insert(
            ColliderBuilder::cuboid(thickness, init_config.sim.height / 2.0)
                .translation(vector![-0.5 * init_config.sim.width - thickness / 2.0, 0.0])
                .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
                .restitution(1.0)
                .friction(0.0)
                .build(),
        ),
    );

    let mut watcher =
        notify::recommended_watcher(reload_config).expect("failed to load file watcher");
    watcher
        .watch(Path::new(CONFIG_PATH), NonRecursive)
        .expect("failed to watch config file");

    type Backend = burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>;
    let device = Default::default();
    let num_inputs = init_config.shark.vision_rays as usize * 3 + 3; // rays * (dist, rot, is_fish) + (x vel, y vel, angular vel)
    let num_outputs = 9; // cardinal directions and neutral (left, left forward, forward, right forward, ...)
    let mut model = dqn::ModelConfig::new(
        num_inputs,
        init_config.dqn.hidden_size as usize,
        num_outputs,
        init_config.dqn.replay_buffer_size as usize,
        init_config.dqn.batch_size as usize,
        init_config.dqn.tau,
        init_config.dqn.gamma,
        init_config.dqn.learning_rate,
        init_config.dqn.eps_decay,
    )
    .init::<Backend>(&device);
    let mut input_data = vec![0.0f32; num_inputs];
    log::debug!("Loading model: {:#?}", model.network);

    let mut rng = Pcg64::from_entropy();
    let (shark_handle, mouth_handle) =
        create_shark(&mut rigid_body_set, &mut collider_set, &mut rng);
    let mut fishes = vec![];
    let mut dead_fishes = vec![];
    let mut sightings = vec![];
    for _ in 0..init_config.fish.count {
        create_fish(
            &mut fishes,
            &mut rigid_body_set,
            &mut collider_set,
            &mut rng,
            &init_config,
        );
    }

    // Create other structures necessary for the simulation.
    let gravity = vector![0.0, 0.0];
    let integration_parameters = IntegrationParameters::default();
    let mut physics_pipeline = PhysicsPipeline::new();
    let mut island_manager = IslandManager::new();
    let mut broad_phase = DefaultBroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut impulse_joint_set = ImpulseJointSet::new();
    let mut multibody_joint_set = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();
    let mut query_pipeline = QueryPipeline::new();
    let physics_hooks = ();
    let event_handler = ();

    let debug_render_mode = DebugRenderMode::COLLIDER_SHAPES;
    let debug_render_style = DebugRenderStyle::default();
    let mut debug_render_pipeline = DebugRenderPipeline::new(debug_render_style, debug_render_mode);
    let mut step = 0;
    let mut human_control = false;
    let mut last_eat_step = 0;
    let mut last_state = None;
    let mut last_action = 0;
    let mut last_reward = 0.0;
    while !rl.window_should_close() {
        let config = load_config();
        for _ in 0..config.sim.sims_per_frame {
            sightings.clear();

            use raylib::consts::KeyboardKey::*;
            if rl.is_key_pressed(KEY_SPACE) {
                human_control = !human_control;
            }

            let (forward, right) = if !human_control {
                if step - last_eat_step > config.dqn.reset_delay as u64 {
                    last_state = None;
                    last_eat_step = step;
                    log::info!("Reseting shark due to long time since last eating.");
                    let rot = rng.gen::<f32>() * std::f32::consts::TAU;
                    let shark = rigid_body_set.get_mut(shark_handle).unwrap();
                    shark.set_position(Isometry::new(vector![0.0, 0.0], rot), true);
                }
                let shark_pos = *rigid_body_set.get_mut(shark_handle).unwrap().position();
                // Ray count can not change mid run. Would break the model.
                let ray_count = init_config.shark.vision_rays;
                let vision_angle = config.shark.vision_angle;
                let step_angle = vision_angle / ray_count as f32;
                let start = (step_angle - vision_angle) / 2.0;
                for i in 0..ray_count {
                    let angle = start + step_angle * i as f32;
                    let (sin, cos) = angle.to_radians().sin_cos();
                    let ray = rapier2d::geometry::Ray::new(
                        point![0.0, 0.0],
                        vector![sin, cos] * config.shark.vision_dist,
                    )
                    .transform_by(&shark_pos);

                    if let Some((handle, dist)) = query_pipeline.cast_ray(
                        &rigid_body_set,
                        &collider_set,
                        &ray,
                        1.0,
                        /*solid=*/ true,
                        QueryFilter::default()
                            .exclude_sensors()
                            .exclude_rigid_body(shark_handle),
                    ) {
                        let is_fish = !walls.contains(&handle);
                        let target_pos = collider_set.get_mut(handle).unwrap().position();
                        let target_rot = shark_pos.rotation.rotation_to(&target_pos.rotation);

                        input_data[i as usize * 3 + 0] = dist;
                        input_data[i as usize * 3 + 1] = target_rot.angle();
                        input_data[i as usize * 3 + 2] = (is_fish as u32) as f32;
                        sightings.push(ray.point_at(dist));
                    } else {
                        input_data[i as usize * 3 + 0] = 1.0;
                        input_data[i as usize * 3 + 1] = 0.0;
                        input_data[i as usize * 3 + 2] = 0.0;
                    }
                }
                let state =
                    burn::tensor::Tensor::<Backend, 1>::from_floats(input_data.as_slice(), &device);
                if let Some(last_state) = last_state {
                    let hist = dqn::Hist {
                        state: last_state,
                        action: last_action,
                        reward: last_reward,
                        next_state: state.clone(),
                    };
                    model.push(hist);
                    if step % config.dqn.train_freq as u64 == 0 {
                        model.train(&mut rng);
                    }
                    if step % 10000 == 0 {
                        log::debug!(
                            "step {}: {}, {}",
                            step,
                            model.memory.len(),
                            model.eps_threshold(step)
                        )
                    }
                }
                last_state = Some(state.clone());
                let action = model.pick_action(state, step, &mut rng);
                last_action = action;
                match action {
                    0 => (-1, -1),
                    1 => (-1, 0),
                    2 => (-1, 1),
                    3 => (0, -1),
                    4 => (0, 0),
                    5 => (0, 1),
                    6 => (1, -1),
                    7 => (1, 0),
                    8 => (1, 1),
                    _ => unreachable!(),
                }
            } else {
                last_state = None;
                let forward = rl.is_key_down(KEY_UP) as i32 - rl.is_key_down(KEY_DOWN) as i32;
                let right = rl.is_key_down(KEY_RIGHT) as i32 - rl.is_key_down(KEY_LEFT) as i32;
                // Limit max backwards acceleration.
                (forward, right)
            };

            let forward = (config.shark.max_force * forward as f32)
                .max(config.shark.max_reverse_force)
                .min(config.shark.max_force);

            let shark = rigid_body_set.get_mut(shark_handle).unwrap();
            let shark_forward_force = forward;
            let shark_torque = config.shark.max_torque * right as f32;
            let shark_rot = shark.rotation();
            let shark_force = vector![
                shark_forward_force * -shark_rot.im,
                shark_forward_force * shark_rot.re
            ];
            shark.set_linear_damping(config.shark.linear_damping);
            shark.set_angular_damping(config.shark.angular_damping);

            shark.reset_forces(true);
            shark.add_force(shark_force, true);
            shark.reset_torques(true);
            shark.add_torque(shark_torque, true);

            for fish in fishes.iter() {
                let fish_rigid_body = rigid_body_set.get_mut(fish.handle).unwrap();
                fish_rigid_body.set_linear_damping(config.fish.linear_damping);
                fish_rigid_body.set_angular_damping(config.fish.angular_damping);

                let forward = fish.genome[0] * 2.0 - 1.0;
                let right = fish.genome[1] * 2.0 - 1.0;

                let fish_forward_force = (config.fish.max_force * forward)
                    .max(config.fish.max_reverse_force)
                    .min(config.fish.max_force);

                let fish_torque = config.fish.max_torque * right;
                let fish_rot = fish_rigid_body.rotation();
                let fish_force = vector![
                    fish_forward_force * -fish_rot.im,
                    fish_forward_force * fish_rot.re
                ];

                fish_rigid_body.reset_forces(true);
                fish_rigid_body.add_force(fish_force, true);
                fish_rigid_body.reset_torques(true);
                fish_rigid_body.add_torque(fish_torque, true);
            }

            physics_pipeline.step(
                &gravity,
                &integration_parameters,
                &mut island_manager,
                &mut broad_phase,
                &mut narrow_phase,
                &mut rigid_body_set,
                &mut collider_set,
                &mut impulse_joint_set,
                &mut multibody_joint_set,
                &mut ccd_solver,
                Some(&mut query_pipeline),
                &physics_hooks,
                &event_handler,
            );

            dead_fishes.clear();
            for (collider1, collider2, intersecting) in
                narrow_phase.intersection_pairs_with(mouth_handle)
            {
                if !intersecting {
                    continue;
                }

                let other_collider = if collider1 == mouth_handle {
                    collider2
                } else {
                    collider1
                };
                let fish_handle = collider_set
                    .get_mut(other_collider)
                    .unwrap()
                    .parent()
                    .unwrap();

                let index = rigid_body_set.get(fish_handle).unwrap().user_data;
                dead_fishes.push(index as usize);

                log::debug!("shark ate fish {:?}", index);
            }

            last_reward = dead_fishes.len() as f32;
            if last_reward > 0.0 {
                last_eat_step = step;
            }

            // If extra fish where requested, just add them via extra breeding.
            for _ in fishes.len()..(config.fish.count as usize) {
                let handle = create_fish_rigid_body(&mut rigid_body_set, &mut collider_set);
                let i = fishes.len();
                rigid_body_set.get_mut(handle).unwrap().user_data = i as u128;
                let fish = Fish {
                    handle,
                    genome: [0.5; 2],
                };
                fishes.push(fish);
                dead_fishes.push(i);
            }

            // For each dead fish, have another fish reproduce.
            for &dead_index in dead_fishes.iter() {
                let mut parent_index = rng.gen_range(0..(fishes.len() - dead_fishes.len()));
                while dead_fishes.contains(&parent_index) {
                    parent_index += 1;
                    if parent_index == fishes.len() {
                        parent_index = 0;
                    }
                }
                log::debug!("fish {:?} reproduced", parent_index);

                // TODO: look into smarter mutation and 2 parent crossover.
                let mut parent_pos = *rigid_body_set
                    .get(fishes[parent_index].handle)
                    .unwrap()
                    .position();

                fishes[dead_index].genome = fishes[parent_index].genome;
                for g in fishes[dead_index].genome.iter_mut() {
                    let shift = rng.gen::<f32>() * config.fish.mutation_factor
                        - config.fish.mutation_factor / 2.0;
                    *g = sig(inv_sig(*g) + shift);
                }

                let rot = rng.gen::<f32>() * std::f32::consts::TAU;
                parent_pos.rotation = Rotation::new(rot);
                rigid_body_set
                    .get_mut(fishes[dead_index].handle)
                    .unwrap()
                    .set_position(parent_pos, true);
            }

            // Remove any extra fish
            for i in (config.fish.count as usize)..fishes.len() {
                rigid_body_set.remove(
                    fishes[i].handle,
                    &mut island_manager,
                    &mut collider_set,
                    &mut impulse_joint_set,
                    &mut multibody_joint_set,
                    true,
                );
            }
            fishes.truncate(config.fish.count as usize);
            step += 1;
        }

        let fps = rl.get_fps().to_string();
        let width = rl.get_screen_width();
        let height = rl.get_screen_height();
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::DIMGRAY);

        let mut render_backend = DebugRaylibRender::new(&mut d, width, height, &config.sim);
        debug_render_pipeline.render(
            &mut render_backend,
            &rigid_body_set,
            &collider_set,
            &impulse_joint_set,
            &multibody_joint_set,
            &narrow_phase,
        );

        render_backend.draw_sightings(&sightings);
        d.draw_text(&fps, width - 35, 10, 20, Color::LINEN.alpha(0.5));
    }
}

fn create_shark(
    rigid_body_set: &mut RigidBodySet,
    collider_set: &mut ColliderSet,
    rng: &mut Pcg64,
) -> (RigidBodyHandle, ColliderHandle) {
    let rigid_body = RigidBodyBuilder::dynamic().build();
    let shark_handle = rigid_body_set.insert(rigid_body);
    let body = ColliderBuilder::triangle(point![0.0, 4.0], point![-1.0, 0.0], point![1.0, 0.0])
        .restitution(1.0)
        .friction(0.0)
        .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
        .build();
    let mouth = ColliderBuilder::triangle(point![0.0, 3.5], point![-0.5, 4.5], point![0.5, 4.5])
        .collision_groups(InteractionGroups::new(EDIBLE_GROUP, EDIBLE_GROUP))
        .mass(0.0)
        .sensor(true);

    let rot = rng.gen::<f32>() * std::f32::consts::TAU;
    let shark = rigid_body_set.get_mut(shark_handle).unwrap();
    shark.set_position(Isometry::new(vector![0.0, 0.0], rot), true);

    collider_set.insert_with_parent(body, shark_handle, rigid_body_set);
    let mouth_handle = collider_set.insert_with_parent(mouth, shark_handle, rigid_body_set);
    (shark_handle, mouth_handle)
}

struct Fish {
    handle: RigidBodyHandle,
    genome: [f32; 2],
}

fn create_fish(
    fishes: &mut Vec<Fish>,
    rigid_body_set: &mut RigidBodySet,
    collider_set: &mut ColliderSet,
    rng: &mut Pcg64,
    config: &Config,
) {
    let handle = create_fish_rigid_body(rigid_body_set, collider_set);
    randomize_position(rigid_body_set, handle, rng, config);
    let i = fishes.len();
    rigid_body_set.get_mut(handle).unwrap().user_data = i as u128;

    // Start with super small non-neutral genome.
    // Neutral is 0.5.
    let mut fish = Fish {
        handle,
        genome: [0.5; 2],
    };
    let scale = 0.05;
    for g in fish.genome.iter_mut() {
        *g += rng.gen::<f32>() * scale - scale / 2.0;
    }
    fishes.push(fish);
}

fn create_fish_rigid_body(
    rigid_body_set: &mut RigidBodySet,
    collider_set: &mut ColliderSet,
) -> RigidBodyHandle {
    let rigid_body = RigidBodyBuilder::dynamic().build();
    let body = ColliderBuilder::triangle(point![0.0, 2.0], point![-0.5, 0.0], point![0.5, 0.0])
        .restitution(1.0)
        .friction(0.0)
        .collision_groups(InteractionGroups::new(
            EDIBLE_GROUP | MAIN_GROUP,
            EDIBLE_GROUP | MAIN_GROUP,
        ))
        .build();
    let fish_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(body, fish_handle, rigid_body_set);
    fish_handle
}

fn randomize_position(
    rigid_body_set: &mut RigidBodySet,
    handle: RigidBodyHandle,
    rng: &mut Pcg64,
    config: &Config,
) {
    let x = (rng.gen::<f32>() - 0.5) * (config.sim.width - 0.5);
    let y = (rng.gen::<f32>() - 0.5) * (config.sim.height - 1.0);
    let rot = rng.gen::<f32>() * std::f32::consts::TAU;
    log::trace!("Placing randomly at ({}, {}) with rotation {}", x, y, rot);
    let obj = rigid_body_set.get_mut(handle).unwrap();
    obj.set_position(Isometry::new(vector![x, y], rot), true);
}

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn inv_sig(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

struct DebugRaylibRender<'a, 'b> {
    d: &'a mut RaylibDrawHandle<'b>,
    scale_w: f32,
    offset_w: f32,
    scale_h: f32,
    offset_h: f32,
}

impl<'a, 'b> DebugRaylibRender<'a, 'b> {
    fn new(
        d: &'a mut RaylibDrawHandle<'b>,
        screen_width: i32,
        screen_height: i32,
        sim: &config::Sim,
    ) -> Self {
        let scale_w = screen_width as f32 / sim.width;
        let scale_h = screen_height as f32 / sim.height;
        let offset_w = screen_width as f32 / 2.0;
        let offset_h = screen_height as f32 / 2.0;
        Self {
            d,
            scale_w,
            scale_h,
            offset_w,
            offset_h,
        }
    }

    fn scale_point(self: &Self, point: Point<f32>) -> Vector2 {
        Vector2::new(
            point.x * self.scale_w + self.offset_w,
            point.y * self.scale_h + self.offset_h,
        )
    }

    fn draw_sightings(&mut self, sightings: &[Point<f32>]) {
        for &point in sightings {
            let point = self.scale_point(point);
            self.d
                .draw_circle_v(point, 4.0, Color::DARKSALMON.alpha(0.8));
        }
    }
}

impl DebugRenderBackend for DebugRaylibRender<'_, '_> {
    fn draw_line(
        &mut self,
        _object: DebugRenderObject<'_>,
        a: Point<f32>,
        b: Point<f32>,
        color: [f32; 4],
    ) {
        let a = self.scale_point(a);
        let b = self.scale_point(b);
        let c = Color::color_from_normalized(Vector4::new(color[0], color[1], color[2], color[2]));
        self.d.draw_line_ex(a, b, 4.0, c);
    }
}

const CONFIG_PATH: &'static str = "config.toml";
static CONFIG: Mutex<MaybeUninit<Config>> = Mutex::new(MaybeUninit::uninit());
static CONFIG_UPDATED: AtomicBool = AtomicBool::new(false);

fn load_config_from_file() -> Config {
    let config = std::fs::read_to_string(CONFIG_PATH).expect("failed to read config");
    let out = toml::from_str(&config).expect("failed to deserialize config");
    log::debug!("Loaded config: {:#?}", out);
    out
}

fn init_config() {
    let config = load_config_from_file();
    CONFIG.lock().unwrap().write(config);
}

fn reload_config(res: notify::Result<notify::Event>) {
    match res {
        Ok(_) if !CONFIG_UPDATED.load(Ordering::Acquire) => {
            CONFIG_UPDATED.store(true, Ordering::Release);
            let config = load_config_from_file();
            log::info!("Loaded new config");
            CONFIG.lock().unwrap().write(config);
        }
        Ok(_) => {}
        Err(e) => panic!("Failed to watch config file: {:?}", e),
    }
}

fn load_config() -> Config {
    CONFIG_UPDATED.store(false, Ordering::Release);
    unsafe { CONFIG.lock().unwrap().assume_init() }
}
