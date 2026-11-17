import asyncio
import numpy as np
import warp as wp
import pinocchio as pin
import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.pink import PinkIKController, load_pink_robot

# Load PINK model
pink_robot = load_pink_robot(
    urdf_path="/home/spiessenberger/isaacsim/test_repo/kuka.urdf",
    package_dirs=["/home/spiessenberger/isaacsim/test_repo/meshes"],
    srdf_path="/home/spiessenberger/isaacsim/test_repo/iiwa7.srdf",
    build_collision_model=True,
)

# --- Programmatically add the virtual TCP frame ---
parent_joint_name = "iiwa7_joint_7"  # Adjust if your last arm joint has a different name
parent_joint_id = pink_robot.model.getJointId(parent_joint_name)

# Adjust this translation [x, y, z] in meters to position the TCP between the finger pads
tcp_offset = np.array([0.0, 0.0, 0.12]) 
tcp_placement = pin.SE3(np.eye(3), tcp_offset)

tool_link_name = "robotiq_tcp"
pink_robot.model.addFrame(
    pin.Frame(
        tool_link_name,
        parent_joint_id,
        0,  # parent_frame_id (0 for root of joint frame)
        tcp_placement,
        pin.FrameType.OP_FRAME
    )
)
# Recreate Pinocchio workspace data to include the new frame
pink_robot.data = pink_robot.model.createData()

async def initialize_and_run():
    # Create world and find objects
    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)

    # Initialize the existing simulation context
    await world.initialize_simulation_context_async()

    # Initialize physics
    world.initialize_physics()

    print("Physics initialized!")
    print("Physics context:", world.get_physics_context())
    
    if world.scene.get_object("kuka") == False:
        robot = world.scene.add(SingleArticulation(prim_path="/World/kuka", name="kuka")) 
    articulation = world.scene.get_object("kuka")
    if articulation is None:
        raise RuntimeError("KUKA not found in scene!")
    
    # Initialize only if it hasn't already been initialized
    articulation.initialize()

    robot_joint_space = articulation.dof_names
    robot_site_space = [tool_link_name]

    print("Joints:", robot_joint_space)
    print("Joint positions:", articulation.get_joint_positions())

    target_position = wp.array(
        [[0.4, 0.2, 0.5]],
        dtype=wp.float32
    )

    target_orientation = wp.array(
        [[0.0, 0.0, -1.0, 0.0]],
        dtype=wp.float32
    )

    def create_target(pos, orient):
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=robot_site_space,
                positions=(robot_site_space, pos),
                orientations=(robot_site_space, orient),
            )
        )

    def get_robot_state():
        positions = wp.array(
            np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float32
            ).reshape(-1),
            dtype=wp.float32
        )

        velocities = wp.array(
            np.asarray(
                articulation.get_joint_velocities(),
                dtype=np.float32
            ).reshape(-1),
            dtype=wp.float32
        )

        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=robot_joint_space,
                positions=(robot_joint_space, positions),
                velocities=(robot_joint_space, velocities),
            )
        )

    # Create target setpoint state
    setpoint_state = create_target(target_position, target_orientation)

    # Create PINK controller using the virtual TCP frame
    controller = PinkIKController(
        pink_robot=pink_robot,
        robot_joint_space=robot_joint_space,
        robot_site_space=robot_site_space,
        tool_frame=tool_link_name,
        position_cost=1.0,
        orientation_cost=1.0,
        posture_cost=1e-3,
        dt=1.0 / 60.0,
    )

    # Initialize PINK with current positions
    estimated_state = get_robot_state()

    success = controller.reset(
        estimated_state=estimated_state,
        setpoint_state=None,
        t=0.0,
    )

    if not success:
        raise RuntimeError("PINK initialization failed!")

    print("PINK initialized successfully!")

    simulation_time = 0.0

    def on_physics_step(step_size):
        nonlocal simulation_time

        simulation_time += step_size

        try:
            # Read current state
            est_state = get_robot_state()

            # Calculate IK
            desired_state = controller.forward(
                est_state,
                setpoint_state,
                simulation_time,
            )

            if desired_state is not None:
                joint_positions = np.asarray(
                    desired_state.joints.positions.numpy(),
                    dtype=np.float32
                ).reshape(-1)

                joint_indices = np.asarray(
                    desired_state.joints.position_indices.numpy(),
                    dtype=np.int32
                ).reshape(-1)

                # Apply action
                action = ArticulationAction(
                    joint_positions=joint_positions,
                    joint_indices=joint_indices
                )

                articulation.apply_action(action)

        except Exception:
            import traceback
            traceback.print_exc()
            raise

    # Remove previous callback if the script was run before
    try:
        world.remove_physics_callback("pink_ik_controller")
    except Exception:
        pass

    world.add_physics_callback(
        "pink_ik_controller",
        callback_fn=on_physics_step,
    )

    world.play()
    print("PINK IK controller running!")


# Start asynchronous initialization
asyncio.ensure_future(initialize_and_run())