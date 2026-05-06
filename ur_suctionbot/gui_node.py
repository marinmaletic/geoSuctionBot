#!/usr/bin/env python3
"""
gui_node

Simple tkinter GUI for controlling suction scoring.
Run standalone: ros2 run ur_suctionbot gui_node.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from ur_suctionbot.srv import ComputeSuction, Segment


class GuiNode(Node):
    def __init__(self):
        super().__init__('gui_node')

        self._compute_client = self.create_client(ComputeSuction, '/ur_suctionbot/suction/compute')
        self._save_client = self.create_client(Trigger, '/ur_suctionbot/camera/save_frame')
        self._segment_client = self.create_client(Segment, '/ur_suctionbot/segmentation/segment')

    def check_nodes(self):
        names = [n for n, _ in self.get_node_names_and_namespaces()]
        return 'camera_node' in names, 'suction_node' in names


class SuctionGUI:
    def __init__(self, node: GuiNode):
        self._node = node

        self._root = tk.Tk()
        self._root.title('ur_suctionbot — Suction Control')
        self._root.resizable(False, False)

        pad = {'padx': 8, 'pady': 4}

        # ── Node status ──────────────────────────────────────────
        status_frame = ttk.LabelFrame(self._root, text='Node Status')
        status_frame.pack(fill='x', **pad)

        self._lbl_camera  = tk.Label(status_frame, text='Camera:  ●',
                                     font=('monospace', 10), fg='gray')
        self._lbl_suction = tk.Label(status_frame, text='Suction: ●',
                                     font=('monospace', 10), fg='gray')
        self._lbl_camera.pack(anchor='w', padx=8)
        self._lbl_suction.pack(anchor='w', padx=8)

        # ── Method ───────────────────────────────────────────────
        method_frame = ttk.LabelFrame(self._root, text='Scoring Method')
        method_frame.pack(fill='x', **pad)

        self._method = tk.StringVar(value='knn')
        for m in ('knn', 'sobel', 'ransac'):
            ttk.Radiobutton(method_frame, text=m.upper(),
                            variable=self._method, value=m).pack(
                side='left', padx=8, pady=4)

        # ── Parameters ───────────────────────────────────────────
        param_frame = ttk.LabelFrame(self._root, text='Parameters')
        param_frame.pack(fill='x', **pad)

        ttk.Label(param_frame, text='Threshold (0-1)').grid(
            row=0, column=0, sticky='w', padx=8, pady=2)
        self._threshold = tk.DoubleVar(value=0.5)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._threshold, width=8).grid(
            row=0, column=1, padx=8, pady=2)

        ttk.Label(param_frame, text='Cup diameter (mm)').grid(
            row=1, column=0, sticky='w', padx=8, pady=2)
        self._cup = tk.DoubleVar(value=30.0)
        ttk.Spinbox(param_frame, from_=0.0, to=100.0, increment=5.0,
                    textvariable=self._cup, width=8).grid(
            row=1, column=1, padx=8, pady=2)

        # ── Buttons ──────────────────────────────────────────────
        btn_frame = tk.Frame(self._root)
        btn_frame.pack(fill='x', **pad)

        self._btn_compute = tk.Button(
            btn_frame, text='Compute Suction Scores',
            bg='#1a6e28', fg='white', font=('sans', 10, 'bold'),
            command=self._on_compute)
        self._btn_compute.pack(side='left', expand=True, fill='x', padx=4)

        self._btn_save = tk.Button(
            btn_frame, text='Save Frame',
            bg='#1a4a6e', fg='white', font=('sans', 10, 'bold'),
            command=self._on_save)
        self._btn_save.pack(side='left', expand=True, fill='x', padx=4)

        self._btn_segment = tk.Button(
            btn_frame, text='Segment Objects',
            bg='#6e1a4a', fg='white', font=('sans', 10, 'bold'),
            command=self._on_segment)
        self._btn_segment.pack(side='left', expand=True, fill='x', padx=4)

        # ── Result ───────────────────────────────────────────────
        result_frame = ttk.LabelFrame(self._root, text='Result')
        result_frame.pack(fill='x', **pad)

        self._lbl_score    = tk.Label(result_frame, text='Best score:  —',
                                      font=('monospace', 10))
        self._lbl_position = tk.Label(result_frame, text='Position:    —',
                                      font=('monospace', 10))
        self._lbl_method   = tk.Label(result_frame, text='Method:      —',
                                      font=('monospace', 10))
        self._lbl_save     = tk.Label(result_frame, text='Save:        —',
                                      font=('monospace', 10))

        for lbl in (self._lbl_score, self._lbl_position,
                    self._lbl_method, self._lbl_save):
            lbl.pack(anchor='w', padx=8, pady=1)

        # ── Status bar ───────────────────────────────────────────
        self._status_bar = tk.Label(self._root, text='Ready',
                                    bd=1, relief='sunken', anchor='w')
        self._status_bar.pack(fill='x', side='bottom')

        # Start node status polling
        self._poll_status()

    def _set_status(self, text):
        self._status_bar.config(text=text)

    def _poll_status(self):
        camera_ok, suction_ok = self._node.check_nodes()
        self._lbl_camera.config(
            text=f'Camera:  ●',
            fg='#00cc44' if camera_ok else '#cc2200')
        self._lbl_suction.config(
            text=f'Suction: ●',
            fg='#00cc44' if suction_ok else '#cc2200')
        self._root.after(1000, self._poll_status)

    def _on_compute(self):
        if not self._node._compute_client.service_is_ready():
            self._set_status('Suction service not available')
            return

        self._btn_compute.config(state='disabled')
        self._set_status('Computing...')

        req = ComputeSuction.Request()
        req.method          = self._method.get()
        req.threshold       = float(self._threshold.get())
        req.cup_diameter_mm = float(self._cup.get())

        threading.Thread(
            target=self._call_compute, args=(req,), daemon=True).start()

    def _call_compute(self, req):
        future = self._node._compute_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
        self._root.after(0, self._on_compute_done, future)

    def _on_compute_done(self, future):
        try:
            response = future.result()
            if response.success:
                p = response.best_candidate.pose.pose.position
                self._lbl_score.config(
                    text=f'Best score:  {response.best_candidate.score:.3f}')
                self._lbl_position.config(
                    text=f'Position:    x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} m')
                self._lbl_method.config(
                    text=f'Method:      {response.best_candidate.method}')
                self._set_status('Done')
            else:
                self._set_status(f'Error: {response.message}')
        except Exception as e:
            self._set_status(f'Failed: {e}')
        finally:
            self._btn_compute.config(state='normal')

    def _on_save(self):
        if not self._node._save_client.service_is_ready():
            self._set_status('Save service not available')
            return

        self._btn_save.config(state='disabled')
        threading.Thread(target=self._call_save, daemon=True).start()

    def _call_save(self):
        future = self._node._save_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)
        self._root.after(0, self._on_save_done, future)

    def _on_save_done(self, future):
        try:
            response = future.result()
            self._lbl_save.config(text=f'Save: {response.message}')
            self._set_status('Frame saved')
        except Exception as e:
            self._set_status(f'Save failed: {e}')
        finally:
            self._btn_save.config(state='normal')

    def _on_segment(self):
        if not self._node._segment_client.service_is_ready():
            self._set_status('Segmentation service not available')
            return

        self._btn_segment.config(state='disabled')
        self._set_status('Segmenting...')

        threading.Thread(target=self._call_segment, daemon=True).start()

    def _call_segment(self):
        from ur_suctionbot.srv import Segment
        req = Segment.Request()
        req.prompt = ''  # use default prompt from node parameter
        future = self._node._segment_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=30.0)
        self._root.after(0, self._on_segment_done, future)

    def _on_segment_done(self, future):
        try:
            response = future.result()
            if response.success:
                self._set_status(f'Segmented: {response.message}')
            else:
                self._set_status(f'Segment error: {response.message}')
        except Exception as e:
            self._set_status(f'Segment failed: {e}')
        finally:
            self._btn_segment.config(state='normal')

    def run(self):
        self._root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = GuiNode()

    # Spin ROS2 in background thread
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    gui = SuctionGUI(node)
    try:
        gui.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
