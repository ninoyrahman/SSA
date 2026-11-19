"""Small Tkinter interface for running maneuver tests interactively."""

import contextlib
import io
import traceback
import tkinter as tk
from tkinter import ttk

import test_maneuvers


class ManeuverTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maneuver Test Runner")
        self.root.geometry("760x500")
        self.test_functions = {
            name: function
            for name, function in vars(test_maneuvers).items()
            if name.startswith("test_") and callable(function)
        }

        controls = ttk.Frame(root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Test:").pack(side=tk.LEFT, padx=(0, 8))
        self.selection = tk.StringVar(value="All tests")
        choices = ["All tests", *self.test_functions.keys()]
        self.dropdown = ttk.Combobox(
            controls,
            textvariable=self.selection,
            values=choices,
            state="readonly",
            width=52,
        )
        self.dropdown.pack(side=tk.LEFT)

        self.run_button = ttk.Button(controls, text="Run", command=self.run_test)
        self.run_button.pack(side=tk.LEFT, padx=(8, 0))

        output_frame = ttk.Frame(root, padding=(10, 0, 10, 10))
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(
            output_frame, orient=tk.VERTICAL, command=self.output.yview
        )
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _write_output(self, message):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, message)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def run_test(self):
        selected = self.selection.get()
        captured = io.StringIO()
        self.run_button.configure(state=tk.DISABLED)
        self._write_output(f"\n> Running {selected}\n")
        self.root.update_idletasks()

        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                if selected == "All tests":
                    test_maneuvers.run_all_tests()
                else:
                    self.test_functions[selected]()
                    print(f"PASS {selected}")
        except Exception:
            captured.write(traceback.format_exc())
        finally:
            result = captured.getvalue()
            self._write_output(result if result else "Completed with no output.\n")
            self.run_button.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    ManeuverTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
