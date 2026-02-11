# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from typing import Optional, Self

import numpy as np
import numpy.typing as npt
from striplog import Striplog

from .Curve import Curve
from .DepthAgeModel import DepthAgeModel
from .Marker import Marker


class Well:
    def __init__(
        self: Self,
        name: str,
        wellHeadCoords: npt.NDArray[np.float64],
        depth: float,
    ) -> None:
        """Defines a well.

        :param str name: name of the well.
        :param npt.NDArray[np.float64] wellHeadCoords: x,y,z coordinates of the
            well head.
        :param float depth: depth of the well
        """
        #: name of the well
        self.name: str = name
        #: x,y,z coordinates of the well head
        self.wellHeadCoords: npt.NDArray[np.float64] = wellHeadCoords
        #: well depth
        self.depth: float = depth
        #: well path containing x,y,z coordinates of the path. By default,
        #: vertical well from well head
        self._wellPath: npt.NDArray[np.float64] = np.array(
            wellHeadCoords - np.array((0.0, 0.0, depth))
        )
        #: well markers defining remarkable horizons
        self._markers: set[Marker] = set()
        #: well logs (discrete or continuous) in depth domain
        self._logs: dict[str, Curve | Striplog] = {}

        #: depth age model
        self._depthAgeModel: DepthAgeModel | None = None
        #: well logs (discrete or continuous) in age domain
        self._ageLogs: dict[str, Curve | Striplog] = {}

    def shallowCopy(
        self: Self, newName: str, copyMarkers: bool, copyLogs: bool
    ) -> "Well":
        """Make a shallow copy of the well.

        Usefull to create a simulated version of the well.

        :param str newName: new name of the well.
        :param bool copyMarkers: if True, copy markers by reference.
        :param bool copyLogs: if True, copy logs by reference.
        :return Well: a copy of the well
        """
        newWell = Well(newName, self.wellHeadCoords, self.depth)
        newWell.setWellPath(self._wellPath)
        if copyMarkers:
            newWell._markers = self._markers
        if copyLogs:
            newWell._logs = self._logs
            newWell._ageLogs = self._ageLogs
        return newWell

    def setWellPath(self: Self, wellPath: npt.NDArray[np.float64]) -> None:
        """Set well path for deviated wells.

        :param npt.NDArray[np.float64] wellPath: array containing
            x,y,z coordinates
        """
        assert wellPath.shape[0] > 1, (
            "Well path must contains at least 2 points."
        )
        assert wellPath.shape[1] == 3, (
            "Well path array must contains 3 columns for x,y,z coordinates"
        )
        self._wellPath = wellPath

    def getMarkers(self: Self) -> set[Marker]:
        """Get well markers.

        :return set[Marker]: set of well markers
        """
        return self._markers

    def setMarkers(self: Self, markers: set[Marker]) -> None:
        """Set well markers.

        Existing makers are deleted. Use `Well.addMarkers` if you want to keep
        existing markers instead.

        :param set[Marker] markers: set of well markers
        """
        self._markers.clear()
        self.addMarkers(markers)

    def addMarkers(self: Self, markers: Marker | set[Marker]) -> None:
        """Add well markers.

        :param Marker | set[Marker] markers: a marker or a set of markers
        """
        if isinstance(markers, Marker):
            self._markers.add(markers)
        else:
            self._markers.update(markers)

    def addLog(self: Self, logName: str, log: Curve | Striplog) -> None:
        """Add a well log.

        Well log can be continuous or discrete. If a log with the name exists,
        it is erased.

        :param str logName: well log name
        :param Curve | Striplog log: input log
        """
        if logName in self._logs:
            print(
                f"WARNING: log {logName} is already in the list of logs, the "
                + "log will be erased."
            )
        if isinstance(log, Striplog):
            self._addDiscreteLog(logName, log)
        elif isinstance(log, Curve):
            self._addContinuousLog(logName, log)
        else:
            print(
                "ERROR: Log type is not managed. Use either Curve or"
                " Striplog types."
            )

    def addAgeLog(self: Self, logName: str, log: Curve | Striplog) -> None:
        """Add a well log in age domain.

        Well log can be continuous or discrete. If a log with the name exists,
        it is erased.

        :param str logName: well log name
        :param Curve | Striplog log: input log
        """
        if logName in self._ageLogs:
            print(
                f"WARNING: age log {logName} is already in the list of age "
                + "logs, the log will be erased."
            )
        self._ageLogs[logName] = log

    def _addDiscreteLog(self: Self, logName: str, log: Striplog) -> None:
        if log.stop.z > self.depth:
            raise ValueError(
                "Well log maximum depth cannot be be greater than well depth."
            )
        self._logs[logName] = log

    def _addContinuousLog(self: Self, logName: str, log: Curve) -> None:
        if log._maxAbscissa > self.depth:
            raise ValueError(
                "Well log maximum depth cannot be be greater than well depth."
            )
        self._logs[logName] = log

    def getDepthLog(self: Self, logName: str) -> Optional[Curve | Striplog]:
        """Get well log in depth domain from its name.

        :param str logName: well log name
        :return Curve | Striplog: output log
        """
        if logName in self._logs:
            return self._logs[logName]
        return None

    def getAgeLog(self: Self, logName: str) -> Optional[Curve | Striplog]:
        """Get well log in depth domain from its name.

        :param str logName: well log name
        :return Curve | Striplog: output log
        """
        if logName in self._ageLogs:
            return self._ageLogs[logName]
        return None

    def getDiscreteLogNames(self: Self) -> set[str]:
        """Get the set of discrete well log names.

        :return set[str]: set of discrete well log names
        """
        logNames: set[str] = set()
        for logName, log in self._logs.items():
            if isinstance(log, Striplog):
                logNames.add(logName)
        return logNames

    def getContinuousLogNames(self: Self) -> set[str]:
        """Get the set of continuous well log names.

        :return set[str]: set of continuous well log names
        """
        logNames: set[str] = set()
        for logName, log in self._logs.items():
            if isinstance(log, Curve):
                logNames.add(logName)
        return logNames

    def initDepthAgeModel(self: Self) -> None:
        """Initialize depth-age model."""
        self._depthAgeModel = DepthAgeModel(self)
        self._depthAgeModel.setMarkers(self._markers)
