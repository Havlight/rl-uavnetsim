from __future__ import annotations

import math

import numpy as np

from rl_uavnetsim import config
from rl_uavnetsim.energy import SimplifiedEnergyModel
from rl_uavnetsim.entities import GroundUser, UAV
from rl_uavnetsim.mobility import RandomWalkMobility


def test_ground_user_backlog_flow_is_closed() -> None:
    user = GroundUser(
        id=1,
        position=np.array([100.0, 200.0, 0.0]),
        velocity=np.zeros(2),
        speed=0.0,
    )

    arrived_bits = user.add_demand_bits(delta_t_s=config.DELTA_T)
    uploaded_bits = user.consume_access_bits(125_000.0)
    user.add_delivered_bits(80_000.0, delta_t_s=config.DELTA_T)

    assert arrived_bits == config.USER_DEMAND_RATE_BPS
    assert uploaded_bits == 125_000.0
    assert user.user_access_backlog_bits == config.USER_DEMAND_RATE_BPS - 125_000.0
    assert user.delivered_bits_step == 80_000.0
    assert user.final_rate_bps == 80_000.0


def test_uav_relay_queue_proportional_dequeue_preserves_bits() -> None:
    uav = UAV(
        id=0,
        position=np.array([0.0, 0.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
    )
    uav.enqueue_relay_bits(11, 300.0)
    uav.enqueue_relay_bits(22, 700.0)

    forwarded_bits_by_user = uav.proportional_dequeue(400.0)

    assert forwarded_bits_by_user[11] == 120.0
    assert forwarded_bits_by_user[22] == 280.0
    assert math.isclose(sum(forwarded_bits_by_user.values()), 400.0)
    assert math.isclose(uav.relay_queue_bits_by_user[11], 180.0)
    assert math.isclose(uav.relay_queue_bits_by_user[22], 420.0)
    assert math.isclose(uav.relay_queue_total_bits, 600.0)


def test_random_walk_moves_ground_user_within_bounds() -> None:
    mobility_model = RandomWalkMobility(
        x_bounds_m=(0.0, 50.0),
        y_bounds_m=(0.0, 50.0),
        speed_mean_mps=5.0,
        speed_max_mps=5.0,
        direction_sigma_rad=0.0,
    )
    user = GroundUser(
        id=2,
        position=np.array([48.0, 25.0, 0.0]),
        velocity=np.array([5.0, 0.0]),
        speed=5.0,
        mobility_model=mobility_model,
    )

    next_position = user.move(delta_t_s=1.0, rng=np.random.default_rng(7))

    assert 0.0 <= next_position[0] <= 50.0
    assert 0.0 <= next_position[1] <= 50.0
    assert next_position[2] == 0.0
    assert math.isclose(np.linalg.norm(user.velocity), user.speed)


def test_uav_action_updates_position_and_energy() -> None:
    uav = UAV(
        id=3,
        position=np.array([10.0, 10.0, config.UAV_HEIGHT]),
        velocity=np.zeros(2),
        speed=0.0,
        direction=0.0,
        energy_model=SimplifiedEnergyModel(),
    )

    next_position, distance_m, energy_used_j = uav.move_by_action(rho_norm=0.5, psi_rad=0.0)

    assert next_position[0] > 10.0
    assert math.isclose(next_position[1], 10.0)
    assert math.isclose(distance_m, 10.0)
    assert math.isclose(energy_used_j, config.E_HOVER + config.E_FLY * 10.0)
    assert math.isclose(uav.residual_energy_j, config.E_INITIAL - energy_used_j)
