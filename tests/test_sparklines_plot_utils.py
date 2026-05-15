import datetime as dt
import unittest

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from app.sparklines_plot_utils import add_tuning_overlay


class AddTuningOverlayTests(unittest.TestCase):
    def test_initial_setpoint_uses_first_average_window_in_xrange(self):
        start = dt.datetime(2026, 5, 7, 7, 36, 10)
        end = start + dt.timedelta(minutes=20)
        sample_times = [
            start + dt.timedelta(seconds=0),
            start + dt.timedelta(seconds=60),
            start + dt.timedelta(seconds=120),
            start + dt.timedelta(seconds=600),
            start + dt.timedelta(seconds=620),
            start + dt.timedelta(seconds=700),
        ]
        values = np.asarray([10.0, 20.0, 30.0, 80.0, 120.0, 100.0])

        fig, ax = plt.subplots()
        try:
            ax.scatter(sample_times, values)
            ax.set_xlim(start, end)

            x_nums = mdates.date2num(sample_times)
            tuning_periods = [(x_nums[3] * 86400.0, x_nums[4] * 86400.0)]
            _, artists = add_tuning_overlay(
                ax,
                tuning_periods=tuning_periods,
                setpoint_avg_window_s=300,
                hide_points=True,
            )

            self.assertEqual(len(artists["steps"]), 1)
            step_line = artists["steps"][0]
            x_data = step_line.get_xdata()
            y_data = step_line.get_ydata()

            self.assertAlmostEqual(x_data[0], mdates.date2num(start))
            self.assertAlmostEqual(x_data[1], x_nums[3])
            self.assertEqual(y_data[0], 20.0)
            self.assertEqual(y_data[1], 20.0)
        finally:
            plt.close(fig)

    def test_none_average_window_keeps_original_tuning_behavior(self):
        start = dt.datetime(2026, 5, 7, 7, 36, 10)
        sample_times = [
            start + dt.timedelta(seconds=0),
            start + dt.timedelta(seconds=600),
            start + dt.timedelta(seconds=620),
            start + dt.timedelta(seconds=700),
        ]
        values = np.asarray([10.0, 80.0, 120.0, 100.0])

        fig, ax = plt.subplots()
        try:
            ax.scatter(sample_times, values)
            ax.set_xlim(start, start + dt.timedelta(minutes=20))

            x_nums = mdates.date2num(sample_times)
            tuning_periods = [(x_nums[1] * 86400.0, x_nums[2] * 86400.0)]
            _, artists = add_tuning_overlay(
                ax,
                tuning_periods=tuning_periods,
                setpoint_avg_window_s=None,
                hide_points=True,
            )

            self.assertEqual(len(artists["steps"]), 1)
            y_data = artists["steps"][0].get_ydata()
            self.assertEqual(y_data[0], 10.0)
            self.assertEqual(y_data[1], 10.0)
            self.assertTrue(np.isnan(y_data[2]))
            self.assertEqual(y_data[3], 120.0)
            self.assertEqual(y_data[4], 120.0)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
