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
    collider_set.insert(
        ColliderBuilder::cuboid(init_config.sim.width / 2.0, thickness)
            .translation(vector![0.0, 0.5 * init_config.sim.height + thickness / 2.0])
            .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
            .restitution(1.0)
            .build(),
    );
    collider_set.insert(
        ColliderBuilder::cuboid(init_config.sim.width / 2.0, thickness)
            .translation(vector![
                0.0,
                -0.5 * init_config.sim.height - thickness / 2.0
            ])
            .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
            .restitution(1.0)
            .build(),
    );
    collider_set.insert(
        ColliderBuilder::cuboid(thickness, init_config.sim.height / 2.0)
            .translation(vector![0.5 * init_config.sim.width + thickness / 2.0, 0.0])
            .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
            .restitution(1.0)
            .build(),
    );
    collider_set.insert(
        ColliderBuilder::cuboid(thickness, init_config.sim.height / 2.0)
            .translation(vector![-0.5 * init_config.sim.width - thickness / 2.0, 0.0])
            .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
            .restitution(1.0)
            .build(),
    );

    let mut watcher =
        notify::recommended_watcher(reload_config).expect("failed to load file watcher");
    watcher
        .watch(Path::new(CONFIG_PATH), NonRecursive)
        .expect("failed to watch config file");

    let (shark_handle, mouth_handle) = create_shark(&mut rigid_body_set, &mut collider_set);
    let mut rng = Pcg64::from_entropy();
    let mut fish_handles = vec![];
    for _ in 0..init_config.fish.count {
        let fish_handle = create_fish(
            &mut rigid_body_set,
            &mut collider_set,
            &mut rng,
            &init_config,
        );
        fish_handles.push(fish_handle);
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
    let physics_hooks = ();
    let event_handler = ();

    let debug_render_mode = DebugRenderMode::COLLIDER_SHAPES;
    let debug_render_style = DebugRenderStyle::default();
    let mut debug_render_pipeline = DebugRenderPipeline::new(debug_render_style, debug_render_mode);
    while !rl.window_should_close() {
        let config = load_config();

        // TODO: only update these on change, not every frame.
        for &fish_handle in fish_handles.iter() {
            let fish = rigid_body_set.get_mut(fish_handle).unwrap();
            fish.set_linear_damping(config.fish.linear_damping);
            fish.set_angular_damping(config.fish.angular_damping);

            let fish_forward_force = (config.fish.max_force * 1.0 as f32)
                .max(config.fish.max_reverse_force)
                .min(config.fish.max_force);

            let fish_torque = config.fish.max_torque * 0.0 as f32;
            let fish_rot = fish.rotation();
            let fish_force = vector![
                fish_forward_force * -fish_rot.im,
                fish_forward_force * fish_rot.re
            ];

            fish.reset_forces(true);
            fish.add_force(fish_force, true);
            fish.reset_torques(true);
            fish.add_torque(fish_torque, true);
        }

        use raylib::consts::KeyboardKey::*;
        let forward = rl.is_key_down(KEY_UP) as i32 - rl.is_key_down(KEY_DOWN) as i32;
        let right = rl.is_key_down(KEY_RIGHT) as i32 - rl.is_key_down(KEY_LEFT) as i32;
        // Limit max backwards acceleration.
        let forward = (config.sharks.max_force * forward as f32)
            .max(config.sharks.max_reverse_force)
            .min(config.sharks.max_force);

        let shark = rigid_body_set.get_mut(shark_handle).unwrap();
        let shark_forward_force = forward;
        let shark_torque = config.sharks.max_torque * right as f32;
        let shark_rot = shark.rotation();
        let shark_force = vector![
            shark_forward_force * -shark_rot.im,
            shark_forward_force * shark_rot.re
        ];
        shark.set_linear_damping(config.sharks.linear_damping);
        shark.set_angular_damping(config.sharks.angular_damping);

        shark.reset_forces(true);
        shark.add_force(shark_force, true);
        shark.reset_torques(true);
        shark.add_torque(shark_torque, true);

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
            None,
            &physics_hooks,
            &event_handler,
        );

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
            let body = collider_set
                .get_mut(other_collider)
                .unwrap()
                .parent()
                .unwrap();
            log::trace!("intersection of shark and {:?}", body);
        }

        let width = rl.get_screen_width();
        let height = rl.get_screen_height();
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::DIMGRAY);

        let mut render_backend = DebugRaylibRender::new(d, width, height, &config.sim);
        debug_render_pipeline.render(
            &mut render_backend,
            &rigid_body_set,
            &collider_set,
            &impulse_joint_set,
            &multibody_joint_set,
            &narrow_phase,
        );
    }
}

fn create_shark(
    rigid_body_set: &mut RigidBodySet,
    collider_set: &mut ColliderSet,
) -> (RigidBodyHandle, ColliderHandle) {
    let rigid_body = RigidBodyBuilder::dynamic().build();
    let body = ColliderBuilder::triangle(point![0.0, 4.0], point![-1.0, 0.0], point![1.0, 0.0])
        .restitution(1.0)
        .collision_groups(InteractionGroups::new(MAIN_GROUP, MAIN_GROUP))
        .build();
    let mouth = ColliderBuilder::triangle(point![0.0, 3.5], point![-0.5, 4.5], point![0.5, 4.5])
        .collision_groups(InteractionGroups::new(EDIBLE_GROUP, EDIBLE_GROUP))
        .sensor(true);
    let shark_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(body, shark_handle, rigid_body_set);
    let mouth_handle = collider_set.insert_with_parent(mouth, shark_handle, rigid_body_set);
    (shark_handle, mouth_handle)
}

fn create_fish(
    rigid_body_set: &mut RigidBodySet,
    collider_set: &mut ColliderSet,
    rng: &mut Pcg64,
    config: &Config,
) -> RigidBodyHandle {
    let rot = rng.gen::<f32>() * std::f32::consts::TAU;
    let x = (rng.gen::<f32>() - 0.5) * (config.sim.width - 0.5);
    let y = (rng.gen::<f32>() - 0.5) * (config.sim.height - 1.0);
    log::trace!("Loading fish at ({}, {})", x, y);
    let rigid_body = RigidBodyBuilder::dynamic()
        .rotation(rot)
        .translation(vector![x, y])
        .build();
    let body = ColliderBuilder::triangle(point![0.0, 2.0], point![-0.5, 0.0], point![0.5, 0.0])
        .restitution(1.0)
        .collision_groups(InteractionGroups::new(
            EDIBLE_GROUP | MAIN_GROUP,
            EDIBLE_GROUP | MAIN_GROUP,
        ))
        .build();
    let fish_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(body, fish_handle, rigid_body_set);
    fish_handle
}

struct DebugRaylibRender<'a> {
    d: RaylibDrawHandle<'a>,
    scale_w: f32,
    offset_w: f32,
    scale_h: f32,
    offset_h: f32,
}

impl<'a> DebugRaylibRender<'a> {
    fn new(
        d: RaylibDrawHandle<'a>,
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
}

impl DebugRenderBackend for DebugRaylibRender<'_> {
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
    log::trace!("Loaded config: {:?}", out);
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
