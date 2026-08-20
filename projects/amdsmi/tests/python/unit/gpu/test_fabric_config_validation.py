#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Fabric ppod/vpod/station config setter unit tests (hardware-free)."""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from common.common import amdsmi

UNSET = 0xFFFFFFFF


@contextlib.contextmanager
def capture(setter_name):
    """Replace a wrapper setter with a stub that records the config it was handed."""
    recorded = []

    def stub(_handle, config_ref):
        recorded.append(config_ref._obj)
        return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

    with mock.patch.object(amdsmi.amdsmi_wrapper, setter_name, stub):
        yield recorded


def handle():
    return amdsmi.amdsmi_wrapper.amdsmi_processor_handle()


class TestFabricPpodConfig(unittest.TestCase):
    def test_mask_only_covers_supplied_fields(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(handle(), accelerator_id=7)

        config = recorded[0]
        self.assertEqual(config.mask, amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID)
        self.assertEqual(config.data.accelerator_id, 7)
        self.assertEqual(config.version, amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_CONFIG_V1)
        self.assertTrue(config.commit)

    def test_mask_accumulates_across_fields(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(
                handle(), accelerator_id=1, ppod_size=2, bandwidth=3, latency=4
            )

        expected = (
            amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_ACCEL_ID
            | amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_PPOD_SIZE
            | amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_BANDWIDTH
            | amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_LATENCY
        )
        self.assertEqual(recorded[0].mask, expected)

    def test_local_accelerator_count_derived_from_list(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(handle(), local_accelerators=[4, 5, 6])

        config = recorded[0]
        self.assertEqual(config.data.local_accelerator_count, 3)
        self.assertEqual(list(config.data.local_accelerators)[:3], [4, 5, 6])
        self.assertEqual(config.mask, amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_PPOD_FIELD_LOCAL_ACCELS)

    def test_ppod_id_requires_exact_length(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config"):
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_gpu_fabric_ppod_config(handle(), ppod_id=[1, 2, 3])

    def test_ppod_id_accepts_full_uuid(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(handle(), ppod_id=list(range(16)))

        self.assertEqual(list(recorded[0].data.ppod_id), list(range(16)))

    def test_bare_commit_sends_empty_mask(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_ppod_config(handle(), commit=True)

        self.assertEqual(recorded[0].mask, 0)
        self.assertTrue(recorded[0].commit)

    def test_rejects_bad_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_ppod_config("not-a-handle", accelerator_id=1)

    def test_rejects_non_integer_value(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config"):
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_gpu_fabric_ppod_config(handle(), accelerator_id="one")

    def test_rejects_oversized_local_accelerators(self):
        with capture("amdsmi_set_gpu_fabric_ppod_config"):
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_gpu_fabric_ppod_config(
                    handle(), local_accelerators=list(range(64))
                )


class TestFabricVpodConfig(unittest.TestCase):
    def test_active_accelerators_pad_with_unset(self):
        with capture("amdsmi_set_gpu_fabric_vpod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(handle(), vpod_active_accelerators=[2, 5])

        accels = list(recorded[0].data.vpod_active_accelerators)
        self.assertEqual(accels[0], 2)
        self.assertEqual(accels[1], 5)
        self.assertTrue(all(slot == UNSET for slot in accels[2:]))

    def test_accelerator_id_zero_is_preserved(self):
        with capture("amdsmi_set_gpu_fabric_vpod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(handle(), vpod_active_accelerators=[0, 1])

        accels = list(recorded[0].data.vpod_active_accelerators)
        self.assertEqual(accels[0], 0)
        self.assertEqual(accels[1], 1)
        self.assertEqual(accels[2], UNSET)

    def test_empty_list_is_all_unset(self):
        with capture("amdsmi_set_gpu_fabric_vpod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(handle(), vpod_active_accelerators=[])

        self.assertTrue(all(slot == UNSET for slot in recorded[0].data.vpod_active_accelerators))

    def test_mask_and_version(self):
        with capture("amdsmi_set_gpu_fabric_vpod_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(handle(), vpod_id=3, addr_mode=1)

        config = recorded[0]
        self.assertEqual(
            config.mask,
            amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_VPOD_FIELD_VPOD_ID
            | amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_VPOD_FIELD_ADDR_MODE,
        )
        self.assertEqual(config.version, amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_VPOD_CONFIG_V1)

    def test_rejects_bad_handle(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_vpod_config(None, vpod_id=1)


class TestFabricStationConfig(unittest.TestCase):
    def test_lane_bitmap_zero_fills_tail(self):
        with capture("amdsmi_set_gpu_fabric_station_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), lane_en_bitmap=[0xFF, 0x0F])

        bitmap = list(recorded[0].data.lane_en_bitmap)
        self.assertEqual(bitmap[0], 0xFF)
        self.assertEqual(bitmap[1], 0x0F)
        self.assertTrue(all(byte == 0 for byte in bitmap[2:]))

    def test_mask_and_version(self):
        with capture("amdsmi_set_gpu_fabric_station_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), station_flags=1, num_stations=2)

        config = recorded[0]
        self.assertEqual(
            config.mask,
            amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_DF_FIELD_STATION_FLAGS
            | amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_DF_FIELD_NUM_STATIONS,
        )
        self.assertEqual(config.data.station_flags, 1)
        self.assertEqual(config.data.num_stations, 2)
        self.assertEqual(config.version, amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_STATION_CONFIG_V1)

    def test_rejects_non_boolean_commit(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), station_flags=1, commit="yes")

    def test_rejects_num_stations_above_byte_range(self):
        with capture("amdsmi_set_gpu_fabric_station_config"):
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), num_stations=300)

    def test_rejects_lane_bitmap_byte_above_range(self):
        with capture("amdsmi_set_gpu_fabric_station_config"):
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), lane_en_bitmap=[0xFF, 256])

    def test_rejects_negative_value(self):
        with capture("amdsmi_set_gpu_fabric_station_config"):
            with self.assertRaises(amdsmi.AmdSmiParameterException):
                amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), station_flags=-1)

    def test_accepts_num_stations_at_byte_maximum(self):
        with capture("amdsmi_set_gpu_fabric_station_config") as recorded:
            amdsmi.amdsmi_set_gpu_fabric_station_config(handle(), num_stations=255)

        self.assertEqual(recorded[0].data.num_stations, 255)


if __name__ == "__main__":
    unittest.main()
