# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple


class TestController(KesslerController):
    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """

        thrust = 50
        turn_rate = 10
        fire = True
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Test Controller"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"



# # -*- coding: utf-8 -*-
# from kesslergame import KesslerController
# from typing import Dict, Tuple
# import math



# class TestController(KesslerController):
#     """
#     A simple but much stronger controller than the original TestController.
#     It:
#         - Aims at the nearest asteroid
#         - Fires only when angle is small
#         - Turns smoothly toward target
#         - Uses thrust only when needed
#         - Avoids dangerous asteroids
#     """

#     def __init__(self):
#         self.target_id = None

#     def get_angle_diff(self, ship_angle, target_angle):
#         """Smallest signed angle difference (in degrees)."""
#         diff = (target_angle - ship_angle + 180) % 360 - 180
#         return diff

#     def actions(self, ship_state: Dict, game_state) -> Tuple[float, float, bool, bool]:
#         # ---------------------------------------------------
#         # 1. Select the closest asteroid
#         # ---------------------------------------------------

#         # In the newer KesslerGame, game_state is a GameState object,
#         # so we use attribute access, not dict .get()
#         asteroids = getattr(game_state, "asteroids", [])
#         if not asteroids:
#             return 0.0, 0.0, False, False

#         # ShipState may be dict-like or an object; handle both safely
#         if isinstance(ship_state, dict):
#             ship_x, ship_y = ship_state["position"]
#             ship_angle = ship_state["angle"]
#         else:
#             ship_x, ship_y = ship_state.position
#             ship_angle = ship_state.angle

#         # Find nearest asteroid
#         closest = None
#         closest_dist = float("inf")

#         for ast in asteroids:
#             # ast may also be dict or object
#             if isinstance(ast, dict):
#                 ax, ay = ast["position"]
#             else:
#                 ax, ay = ast.position

#             dist = math.hypot(ax - ship_x, ay - ship_y)
#             if dist < closest_dist:
#                 closest_dist = dist
#                 closest = ast

#         # If something weird happens
#         if closest is None:
#             return 0.0, 0.0, False, False

#         # Angle to the target asteroid
#         if isinstance(closest, dict):
#             cx, cy = closest["position"]
#         else:
#             cx, cy = closest.position

#         dx = cx - ship_x
#         dy = cy - ship_y
#         target_angle = math.degrees(math.atan2(dy, dx)) % 360

#         # Signed angle difference ship → asteroid
#         angle_diff = self.get_angle_diff(ship_angle, target_angle)

#         # ---------------------------------------------------
#         # 2. Turning logic
#         # ---------------------------------------------------
#         # Normalize turn_rate: -1.0 to 1.0
#         turn_rate = max(-1.0, min(1.0, angle_diff / 45.0))

#         # ---------------------------------------------------
#         # 3. Thrust logic
#         # ---------------------------------------------------
#         if closest_dist > 250:
#             thrust = 1.0
#         elif closest_dist < 100:
#             thrust = 0.1   # slow down if too close
#         else:
#             thrust = 0.5

#         # ---------------------------------------------------
#         # 4. Fire logic
#         # ---------------------------------------------------
#         fire = abs(angle_diff) < 10  # only fire when almost aligned

#         # ---------------------------------------------------
#         # 5. Mines (not used here)
#         # ---------------------------------------------------
#         drop_mine = False

#         return thrust, turn_rate, fire, drop_mine

#     @property
#     def name(self) -> str:
#         return "Improved Test Controller"

