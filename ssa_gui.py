"""Tkinter interface for running util and maneuver tests interactively."""

import contextlib
import io
import inspect
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import test_maneuvers
import test_propagate
import test_util


class TestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SSA Test Runner")
        self.root.geometry("800x520")
        self.gif_path = None
        self.gif_source = None
        self.gif_frame = 0
        self.gif_photo = None
        self.gif_duration = 100
        self.gif_after_id = None

        menu_bar = tk.Menu(root)
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(
            label="Open GIF...",
            command=self.open_gif,
            accelerator="Ctrl+O",
        )
        menu_bar.add_cascade(label="File", menu=file_menu)
        root.configure(menu=menu_bar)
        root.bind("<Control-o>", lambda _event: self.open_gif())
        self.test_groups = {
            "Util": self._get_tests(test_util),
            "Maneuvers": self._get_tests(test_maneuvers),
            "Propagate": self._get_tests(test_propagate),
        }
        action_menu = tk.Menu(menu_bar, tearoff=False)
        action_suites = (
            ("Util", "Util"),
            ("Maneuver", "Maneuvers"),
            ("Propagate", "Propagate"),
        )
        for menu_label, suite_name in action_suites:
            suite_menu = tk.Menu(action_menu, tearoff=False)
            suite_menu.add_command(
                label="All tests",
                command=lambda suite=suite_name:
                    self.run_named_test(suite, "All tests"),
            )
            suite_menu.add_separator()
            for test_name in self.test_groups[suite_name]:
                suite_menu.add_command(
                    label=test_name,
                    command=lambda suite=suite_name, name=test_name:
                        self.run_named_test(suite, name),
                )
            action_menu.add_cascade(label=menu_label, menu=suite_menu)
        menu_bar.add_cascade(label="Action", menu=action_menu)

        self.suite_selection = tk.StringVar(value="Util")
        self.test_selection = tk.StringVar()

        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_tabs = ttk.Notebook(output_frame)
        self.output_tabs.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.Frame(self.output_tabs)
        self.output_tabs.add(text_frame, text="Output")
        self.output = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=self.output.yview
        )
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        gif_frame = ttk.Frame(self.output_tabs)
        self.gif_preview_frame = gif_frame
        self.output_tabs.add(gif_frame, text="Preview")
        self.gif_canvas = tk.Canvas(gif_frame, background="white")
        gif_xscroll = ttk.Scrollbar(
            gif_frame, orient=tk.HORIZONTAL, command=self.gif_canvas.xview
        )
        gif_yscroll = ttk.Scrollbar(
            gif_frame, orient=tk.VERTICAL, command=self.gif_canvas.yview
        )
        self.gif_canvas.configure(
            xscrollcommand=gif_xscroll.set,
            yscrollcommand=gif_yscroll.set,
        )
        self.gif_canvas.bind("<Configure>", self._resize_gif_frame)
        self.gif_canvas.grid(row=0, column=0, sticky="nsew")
        gif_yscroll.grid(row=0, column=1, sticky="ns")
        gif_xscroll.grid(row=1, column=0, sticky="ew")
        gif_frame.rowconfigure(0, weight=1)
        gif_frame.columnconfigure(0, weight=1)

    @staticmethod
    def _get_tests(module):
        return {
            name: function
            for name, function in sorted(vars(module).items())
            if name.startswith("test_") and callable(function)
        }

    def _write_output(self, message):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, message)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def open_gif(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open GIF animation",
            filetypes=(("GIF files", "*.gif"),),
        )
        if not path:
            return

        try:
            image = Image.open(path)
            if image.format != "GIF":
                image.close()
                raise ValueError("The selected file is not a GIF image.")
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "Could not open GIF", str(exc), parent=self.root
            )
            return

        if self.gif_source is not None:
            self.gif_source.close()
        self.gif_source = image
        self.gif_path = path
        self.gif_frame = 0
        self.gif_canvas.delete("all")
        if self.gif_after_id is not None:
            self.root.after_cancel(self.gif_after_id)
            self.gif_after_id = None
        if self._show_gif_frame():
            self.output_tabs.select(self.gif_preview_frame)
            self._animate_gif()

    def _show_gif_frame(self):
        try:
            self.gif_source.seek(self.gif_frame)
        except EOFError:
            if self.gif_frame != 0:
                self.gif_frame = 0
                return self._show_gif_frame()
            return False

        frame = self.gif_source.convert("RGBA")
        self.gif_duration = max(10, int(self.gif_source.info.get("duration", 100)))
        canvas_width = max(1, self.gif_canvas.winfo_width())
        canvas_height = max(1, self.gif_canvas.winfo_height())
        scale = min(
            1.0,
            canvas_width / frame.width,
            canvas_height / frame.height,
        )
        if scale < 1.0:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            frame = frame.resize(
                (max(1, round(frame.width * scale)),
                 max(1, round(frame.height * scale))),
                resampling,
            )

        self.gif_photo = ImageTk.PhotoImage(frame, master=self.root)
        self.gif_canvas.delete("all")
        x = max(0, (canvas_width - self.gif_photo.width()) // 2)
        y = max(0, (canvas_height - self.gif_photo.height()) // 2)
        self.gif_canvas.create_image(x, y, image=self.gif_photo, anchor=tk.NW)
        self.gif_canvas.configure(
            scrollregion=(
                0, 0,
                max(canvas_width, self.gif_photo.width()),
                max(canvas_height, self.gif_photo.height()),
            )
        )
        return True

    def _resize_gif_frame(self, _event=None):
        if self.gif_source is not None:
            self._show_gif_frame()

    def _animate_gif(self):
        if not self.gif_path:
            return
        self.gif_frame += 1
        self._show_gif_frame()
        if self.gif_path:
            self.gif_after_id = self.root.after(
                self.gif_duration, self._animate_gif
            )

    def _run_one(self, name, function):
        captured_output = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured_output), \
                    contextlib.redirect_stderr(captured_output):
                if "root" in inspect.signature(function).parameters:
                    result = function(root=self.root)
                else:
                    result = function()
        finally:
            output = captured_output.getvalue()
            if output:
                self._write_output(output)
                if not output.endswith("\n"):
                    self._write_output("\n")
        self._write_output(f"PASS {name}\n")
        if result is not None:
            self._write_output(f"  Result: {result}\n")

    def run_named_test(self, suite, test_name):
        self.output_tabs.select(0)
        self.suite_selection.set(suite)
        self.test_selection.set(test_name)
        self.run_test()

    def run_test(self):
        suite = self.suite_selection.get()
        selected = self.test_selection.get()
        tests = self.test_groups[suite]
        self._write_output(f"\n> Running {suite}: {selected}\n")
        self.root.update_idletasks()

        if selected == "All tests":
            failures = []
            for name, function in tests.items():
                try:
                    self._run_one(name, function)
                except Exception as exc:
                    failures.append(name)
                    self._write_output(f"FAIL {name}: {exc}\n")
                    self._write_output(traceback.format_exc())
            passed = len(tests) - len(failures)
            self._write_output(f"{passed}/{len(tests)} tests passed.\n")
        else:
            try:
                self._run_one(selected, tests[selected])
            except Exception as exc:
                self._write_output(f"FAIL {selected}: {exc}\n")
                self._write_output(traceback.format_exc())


def main():
    root = tk.Tk()
    TestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
