use rapier2d::prelude::*;
use raylib::prelude::*;

const WIDTH: i32 = 640 * 3 / 2;
const HEIGHT: i32 = 480 * 3 / 2;

const SIM_WIDTH: f32 = 200.0;
const SIM_HEIGHT: f32 = 1.0 / (SCALE_H / HEIGHT as f32);
const SCALE_W: f32 = (WIDTH as f32) / SIM_WIDTH;
const SCALE_H: f32 = SCALE_W * (WIDTH as f32) / (HEIGHT as f32);

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(WIDTH, HEIGHT)
        .title("Sharks and Fish")
        .vsync()
        .build();

    // Use fps to set sim speed.
    // Can run multiple physic steps per frame to run faster.
    // Can also turn of visualization to run super fast.
    rl.set_target_fps(60);

    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    // Create the walls.
    collider_set.insert(
        ColliderBuilder::cuboid(SIM_WIDTH / 2.0, 0.1)
            .translation(vector![0.0, 0.5 * SIM_HEIGHT])
            .restitution(1.0)
            .build(),
    );
    collider_set.insert(
        ColliderBuilder::cuboid(SIM_WIDTH / 2.0, 0.1)
            .translation(vector![0.0, -0.5 * SIM_HEIGHT])
            .restitution(1.0)
            .build(),
    );
    collider_set.insert(
        ColliderBuilder::cuboid(0.1, SIM_HEIGHT / 2.0)
            .translation(vector![0.5 * SIM_WIDTH, 0.0])
            .restitution(1.0)
            .build(),
    );
    collider_set.insert(
        ColliderBuilder::cuboid(0.1, SIM_HEIGHT / 2.0)
            .translation(vector![-0.5 * SIM_WIDTH, 0.0])
            .restitution(1.0)
            .build(),
    );

    // Create the bouncing ball.
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 0.0])
        .build();
    let collider = ColliderBuilder::ball(3.0).restitution(1.0).build();
    let ball_body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, ball_body_handle, &mut rigid_body_set);

    // Create other structures necessary for the simulation.
    let gravity = vector![0.0, 10.0];
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

    let debug_render_mode = DebugRenderMode::all();
    let debug_render_style = DebugRenderStyle::default();
    let mut debug_render_pipeline = DebugRenderPipeline::new(debug_render_style, debug_render_mode);
    while !rl.window_should_close() {
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

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);
        let mut render_backend = DebugRaylibRender(d);
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

struct DebugRaylibRender<'a>(RaylibDrawHandle<'a>);

fn scale_point(point: Point<f32>) -> Vector2 {
    Vector2::new(
        point.x * SCALE_W + (WIDTH / 2) as f32,
        point.y * SCALE_H + (HEIGHT / 2) as f32,
    )
}

impl DebugRenderBackend for DebugRaylibRender<'_> {
    fn draw_line(
        &mut self,
        _object: DebugRenderObject<'_>,
        a: Point<f32>,
        b: Point<f32>,
        color: [f32; 4],
    ) {
        let a = scale_point(a);
        let b = scale_point(b);
        let c = Color::color_from_normalized(Vector4::new(color[0], color[1], color[2], color[2]));
        self.0.draw_line_ex(a, b, 4.0, c);
    }
}
