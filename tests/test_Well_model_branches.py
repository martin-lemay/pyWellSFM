# SPDX-License-Identifier: Apache-2.0
# SPDX-FileContributor: Martin Lemay

from __future__ import annotations

import numpy as np
import pytest
from striplog import Component, Interval, Striplog

from pywellsfm.model.Curve import Curve
from pywellsfm.model.Marker import Marker
from pywellsfm.model.Well import Well
from pywellsfm.utils.logging_utils import clear_stored_logs, get_stored_logs




