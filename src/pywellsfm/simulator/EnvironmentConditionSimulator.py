# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Self

from pywellsfm.model.DepositionalEnvironment import (
    DepositionalEnvironment,
    DepositionalEnvironmentModel,
)


class EnvironmentConditionSimulator:
    def __init__(
        self: Self,
    ) -> None:
        """Simulate environmental conditions based on a conditions model.

        The model is valid for a given environment.
        """
        self.environmentModel: DepositionalEnvironmentModel | None = None

    def setEnvironmentModel(
        self: Self, envModel: DepositionalEnvironmentModel
    ) -> None:
        """Set the environment model used by the simulator.

        :param DepositionalEnvironmentModel envModel: environment model to set.

        :param EnvironmentConditionsModel envConditionsModel: environment
            conditions model to set.
        """
        self.environmentModel = envModel

    def prepare(self: Self) -> None:
        """Prepare the simulator for computations."""
        if self.environmentModel is None:
            raise ValueError("Environment model is not set.")

    def computeEnvironmentalConditions(
        self: Self,
        environmentName: str,
        waterDepth: float,
        age: float,
    ) -> dict[str, float]:
        """Compute environmental conditions for the given environment.

        :param str environmentName: name of the depositional environment for
            the location.
        :param float waterDepth: waterDepth value for the location.
        :param float age: age at which to compute the conditions.

        :return dict[str, float]: dictionary containing environmental
            conditions for the given environment.
        """
        if self.environmentModel is None:
            raise ValueError("Environment model is not set in the simulator.")

        env: DepositionalEnvironment | None = (
            self.environmentModel.getEnvironmentByName(environmentName)
        )
        env_conds: dict[str, float] = {"waterDepth": float(waterDepth)}
        if env is not None:
            env_conds.update(
                env.getEnvironmentConditions(float(waterDepth), age)
            )
        return env_conds
