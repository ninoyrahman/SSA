"""Tkinter interface for running util and maneuver tests interactively."""

import contextlib
import io
import traceback
import tkinter as tk
from tkinter import ttk

import test_maneuvers
import test_propagate
import test_util


class TestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SSA Test Runner")
        self.root.geometry("800x520")
        self.test_groups = {
            "Util": self._get_tests(test_util),
            "Maneuvers": self._get_tests(test_maneuvers),
            "Propagate": self._get_tests(test_propagate),
        }

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Suite:").pack(side=tk.LEFT, padx=(0, 6))
        self.suite_selection = tk.StringVar(value="Util")
        self.suite_dropdown = ttk.Combobox(
            controls,
            textvariable=self.suite_selection,
            values=tuple(self.test_groups),
            state="readonly",
            width=14,
        )
        self.suite_dropdown.pack(side=tk.LEFT, padx=(0, 14))
        self.suite_dropdown.bind("<<ComboboxSelected>>", self._update_test_choices)

        ttk.Label(controls, text="Test:").pack(side=tk.LEFT, padx=(0, 6))
        self.test_selection = tk.StringVar()
        self.test_dropdown = ttk.Combobox(
            controls,
            textvariable=self.test_selection,
            state="readonly",
            width=48,
        )
        self.test_dropdown.pack(side=tk.LEFT)
        self._update_test_choices()

        self.run_button = ttk.Button(controls, text="Run", command=self.run_test)
        self.run_button.pack(side=tk.LEFT, padx=(10, 0))

        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(
            output_frame, orient=tk.VERTICAL, command=self.output.yview
        )
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    @staticmethod
    def _get_tests(module):
        return {
            name: function
            for name, function in sorted(vars(module).items())
            if name.startswith("test_") and callable(function)
        }

    def _update_test_choices(self, _event=None):
        tests = self.test_groups[self.suite_selection.get()]
        choices = ("All tests", *tests.keys())
        self.test_dropdown.configure(values=choices)
        self.test_selection.set(choices[0])

    def _write_output(self, message):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, message)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _run_one(self, name, function):
        result = function()
        self._write_output(f"PASS {name}\n")
        if result is not None:
            self._write_output(f"  Result: {result}\n")

    def run_test(self):
        suite = self.suite_selection.get()
        selected = self.test_selection.get()
        tests = self.test_groups[suite]
        self.run_button.configure(state=tk.DISABLED)
        self._write_output(f"\n> Running {suite}: {selected}\n")
        self.root.update_idletasks()

        failures = []
        try:
            if selected == "All tests":
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
                self._run_one(selected, tests[selected])
        finally:
            self.run_button.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    TestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
