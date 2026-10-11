from smolvla_task.controllers.panda_ik_controller import PandaIKController


class PandaIKControllerNoGravity(PandaIKController):
    """Use the same Panda IK and actuators without bias-force compensation."""

    def __init__(
        self,
        model,
        data,
        ee_site_name="ee_site",
        damping=1e-3,
        step_size=0.5,
        max_joint_step=0.05,
        position_tolerance=0.005,
        orientation_tolerance=0.03,
        orientation_weight=1.0,
    ):
        super().__init__(
            model,
            data,
            ee_site_name=ee_site_name,
            damping=damping,
            step_size=step_size,
            max_joint_step=max_joint_step,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
            orientation_weight=orientation_weight,
            gravity_compensation=False,
        )

    @property
    def gravity_compensation(self):
        return False

    @gravity_compensation.setter
    def gravity_compensation(self, enabled):
        if enabled:
            raise ValueError("PandaIKControllerNoGravity cannot enable compensation.")

    def get_diagnostics(self):
        diagnostics = super().get_diagnostics()
        # The parent reports the theoretical offset even when it is disabled.
        diagnostics["compensation_offset"].fill(0.0)
        return diagnostics
