import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import json
# Standard GUI imports
# Use themed ttk widgets for a more polished, native-looking UI.  We still fall
# back to classic *tk* widgets where colour customisation is required (e.g. the
# dark header bar).
from tkinter import (
    Tk,
    Frame,
    Label,
    Canvas,
    StringVar,
    messagebox,
)
# ``ttk`` ships with a more modern widget set and better styling capabilities
# out of the box, so we prefer it for most controls.
from tkinter import ttk

# The DB layer exposes lightweight ORM-like helpers.  We now also need
# the new ``delete_user`` utility.
from db import SessionLocal, init_db, User, delete_user
from config_utils import load_config, save_config
from audio_utils import Recorder, trim_wav_silence
from audio_embedding import get_audio_embedding


class VoiceAuthApp(Tk):
    def __init__(self):
        super().__init__()

        # ------------------------------------------------------------------
        # General window setup
        # ------------------------------------------------------------------
        self.APP_NAME = "VoiceGuard"  # Public-facing product name

        self.title(f"{self.APP_NAME} – Voice Authentication Demo")
        self.geometry("700x500")
        self.configure(bg="#f0f4f7")  # Light grey/blue background for polish

        # Create a ttk style once and reuse across the app.  This yields a more
        # consistent, platform-native look & feel compared to the classic tk
        # widgets.
        self.style = ttk.Style(self)
        # Use a modern theme where available (falls back gracefully).
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")

        # Global font configuration
        default_font = ("Segoe UI", 11)
        self.style.configure("TLabel", font=default_font, background="#f0f4f7")
        self.style.configure("TButton", font=default_font)
        self.style.configure("TEntry", font=default_font)

        # Dedicated header style
        self.style.configure(
            "Header.TLabel",
            font=("Segoe UI", 20, "bold"),
            foreground="white",
            background="#2c3e50",
        )

        # Init DB (only user table)
        init_db()

        # Remove any leftover audio artefacts from previous runs.
        try:
            from audio_utils import cleanup_recordings_dir

            cleanup_recordings_dir()
        except Exception:
            # Non-fatal – if cleanup fails we continue; the user prefers not to
            # store audio but failure to delete isn’t critical for app logic.
            pass

        self.session = SessionLocal()

        # Load configuration JSON
        self.config = load_config()

        # ------------------------------------------------------------------
        # Layout:   [Header]  ->  [Navigation]  ->  [Dynamic content frame]
        # ------------------------------------------------------------------
        self._build_header()
        self._build_nav()

        # Show default page (Register)
        self._build_register_form()

    def _clear_frame(self):
        # remove all widgets from content_frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    # ------------------------------------------------------------------
    # UI Building blocks
    # ------------------------------------------------------------------

    def _build_header(self):
        """Creates the dark top bar with the product name."""

        self.header_frame = Frame(self, bg="#2c3e50")
        self.header_frame.pack(side="top", fill="x")

        Label(
            self.header_frame,
            text=self.APP_NAME,
            font=("Segoe UI", 20, "bold"),
            bg="#2c3e50",
            fg="white",
        ).pack(side="left", padx=20, pady=15)

        # Optional placeholder for right-hand side future logo / account name

    def _build_nav(self):
        """Horizontal navigation bar with three primary actions."""

        self.nav_frame = Frame(self, bg="#34495e")
        self.nav_frame.pack(side="top", fill="x")

        # Track nav buttons to allow highlighting of the active tab later.
        self.nav_buttons = {}

        # Helper to create nav buttons with consistent styling, storing them
        # for later highlighting.
        def nav_button(text: str, command):
            btn = ttk.Button(
                self.nav_frame,
                text=text,
                command=command,
                style="Nav.TButton",
            )
            self.nav_buttons[text] = btn
            return btn

        # Base style for navigation items (dark bar)
        self.style.configure(
            "Nav.TButton",
            background="#34495e",
            foreground="white",
            borderwidth=0,
            focusthickness=3,
            focuscolor="none",
        )
        self.style.map(
            "Nav.TButton",
            background=[("active", "#3d566e")],
            foreground=[("disabled", "#bdc3c7"), ("active", "white")],
        )

        # Active tab style – subtle accent colour
        self.style.configure(
            "NavActive.TButton",
            background="#1abc9c",
            foreground="white",
            borderwidth=0,
        )
        self.style.map(
            "NavActive.TButton",
            background=[("active", "#16a085")],
        )

        nav_button("Register", self._build_register_form).pack(side="left", padx=10, pady=8)
        nav_button("Verify", self._build_verify_form).pack(side="left", padx=10, pady=8)
        nav_button("Admin", self._build_admin_panel).pack(side="left", padx=10, pady=8)

        # Utility to switch highlighting
        def set_active(tab_name: str):
            for name, button in self.nav_buttons.items():
                style = "NavActive.TButton" if name == tab_name else "Nav.TButton"
                button.configure(style=style)

        # Expose setter for page builders
        self._set_active_tab = set_active

        # Dedicated frame that holds all page-specific widgets
        self.content_frame = Frame(self, bg="#f0f4f7")
        self.content_frame.pack(expand=True, fill="both", padx=10, pady=10)

    # Landing page
    def _build_landing(self):
        # With the new nav bar, landing page is simply the Register tab
        self._build_register_form()

    # Registration
    def _build_register_form(self):
        # Highlight active navigation tab
        if hasattr(self, "_set_active_tab"):
            self._set_active_tab("Register")

        self._clear_frame()

        ttk.Label(
            self.content_frame,
            text="Registration",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=10)

        self.reg_first_name = StringVar()
        self.reg_last_name = StringVar()
        self.reg_dob = StringVar()

        ttk.Label(self.content_frame, text="First Name:").pack(anchor="w")
        ttk.Entry(self.content_frame, textvariable=self.reg_first_name, width=40).pack(
            pady=2, fill="x"
        )

        ttk.Label(self.content_frame, text="Last Name:").pack(anchor="w")
        ttk.Entry(self.content_frame, textvariable=self.reg_last_name, width=40).pack(
            pady=2, fill="x"
        )

        ttk.Label(self.content_frame, text="Date of Birth (YYYY-MM-DD):").pack(anchor="w")
        ttk.Entry(self.content_frame, textvariable=self.reg_dob, width=40).pack(
            pady=2, fill="x"
        )

        ttk.Button(
            self.content_frame,
            text="Next",  # The user first fills out the form, then continues
            command=self._registration_instructions,
            width=20,
        ).pack(pady=15)

        # Force immediate render on macOS
        self.update_idletasks()

    # ------------------------------------------------------------
    # Registration workflow step 2 –  instructions & “Ready”
    # ------------------------------------------------------------

    def _registration_instructions(self):
        """Show instructions before the actual registration recording starts."""

        # Validate form inputs first and bail out early on bad data
        try:
            datetime.strptime(self.reg_dob.get(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD.")
            return

        self._clear_frame()

        ttk.Label(
            self.content_frame,
            text="Voice Sample Registration",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=(0, 10))

        instructions = (
            "When you click the ‘Ready’ button your microphone will begin "
            "recording.\n\n"
            "A short script will then appear – please read it clearly and "
            "naturally.\n\n"
            "Click ‘Done’ once you have finished reading the script."
        )

        # Ensure text is readable even when the system is in dark-mode where
        # the default label foreground is white.
        Label(
            self.content_frame,
            text=instructions,
            wraplength=550,
            justify="left",
            bg="#f0f4f7",
            fg="#2c3e50",
        ).pack(pady=6)

        ttk.Button(
            self.content_frame,
            text="Ready",
            command=self._start_registration_recording,
            width=20,
        ).pack(pady=15)

    # ------------------
    # Registration flow – variable-length recording
    # ------------------

    def _start_registration_recording(self):
        """Clear the UI, display the script, and start recording."""

        # Prepare UI for live recording
        self._clear_frame()

        script = self.config.get("registration_script", "")

        ttk.Label(
            self.content_frame,
            text="Please read the following script:",
        ).pack(pady=(0, 6))

        Label(
            self.content_frame,
            text=script,
            wraplength=550,
            fg="#2c3e50",
            bg="#f0f4f7",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=4)

        # Start microphone capture as soon as the script is visible
        self.recorder = Recorder()
        self.recorder.start()

        # Show recording status label so the user knows we are capturing
        self._rec_status_label = ttk.Label(self.content_frame, text="Recording…")
        self._rec_status_label.pack(pady=5)

        done_btn = ttk.Button(
            self.content_frame,
            text="Done",
            width=25,
        )
        done_btn.configure(command=lambda b=done_btn: self._finish_registration_recording(b))
        done_btn.pack(pady=10)

    def _finish_registration_recording(self, done_btn):
        """Stop recording, process audio and persist new user."""

        done_btn.configure(state="disabled")
        if self._rec_status_label:
            self._rec_status_label.configure(text="Processing…")

        try:
            wav_path = self.recorder.stop_and_save()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save recording: {exc}")
            return

        def worker():
            try:
                # Trim leading/trailing silence before embedding
                wav_trimmed = trim_wav_silence(wav_path)
                embedding = get_audio_embedding(wav_trimmed)

                # Clean up audio files – we only keep the embedding vector.
                try:
                    wav_trimmed.unlink(missing_ok=True)
                    if wav_trimmed != wav_path:
                        wav_path.unlink(missing_ok=True)
                except Exception:
                    # Non-fatal: if deletion fails (e.g. file still open), we
                    # simply ignore.  The user requested best-effort cleanup.
                    pass

                user = User(
                    first_name=self.reg_first_name.get(),
                    last_name=self.reg_last_name.get(),
                    date_of_birth=datetime.strptime(self.reg_dob.get(), "%Y-%m-%d").date(),
                    embedding=json.dumps(embedding),
                )
                self.session.add(user)
                self.session.commit()

                self.after(
                    0,
                    lambda: [
                        messagebox.showinfo("Success", "Registration completed!"),
                        self._build_landing(),
                    ],
                )
            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # Verification
    def _build_verify_form(self):
        if hasattr(self, "_set_active_tab"):
            self._set_active_tab("Verify")

        self._clear_frame()
        ttk.Label(
            self.content_frame,
            text="Verification",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=10)

        # Show instructions & a **Ready** button. The actual script will be
        # displayed only after the user signals they are prepared, mirroring
        # the new registration flow.

        ttk.Label(
            self.content_frame,
            text="Instructions",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=5)

        verify_instructions = (
            "Press ‘Ready’ to begin verification. Your microphone will start "
            "recording automatically.\n\nRead the script that appears on the "
            "screen out loud, then click ‘Done’."
        )

        # Same readability fix as the registration instructions.
        Label(
            self.content_frame,
            text=verify_instructions,
            wraplength=550,
            justify="left",
            bg="#f0f4f7",
            fg="#2c3e50",
        ).pack(pady=6)

        ttk.Button(
            self.content_frame,
            text="Ready",
            command=self._start_verification_recording,
            width=25,
        ).pack(pady=15)

        self.update_idletasks()

    # ------------------
    # Verification flow – variable-length recording
    # ------------------

    def _start_verification_recording(self):
        """Clear the frame, show the script, and start capturing audio."""

        self._clear_frame()

        script = self.config.get("verification_script", "")

        ttk.Label(
            self.content_frame,
            text="Please read the following script:",
        ).pack(pady=(0, 6))

        Label(
            self.content_frame,
            text=script,
            fg="#2c3e50",
            wraplength=550,
            bg="#f0f4f7",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=4)

        # Start mic
        self.recorder = Recorder()
        self.recorder.start()

        self._rec_status_label = ttk.Label(self.content_frame, text="Recording…")
        self._rec_status_label.pack(pady=5)

        done_btn = ttk.Button(
            self.content_frame,
            text="Done",
            width=25,
        )
        done_btn.configure(command=lambda b=done_btn: self._finish_verification_recording(b))
        done_btn.pack(pady=10)

    def _finish_verification_recording(self, done_btn):
        done_btn.configure(state="disabled")
        if self._rec_status_label:
            self._rec_status_label.configure(text="Processing…")

        try:
            wav_path = self.recorder.stop_and_save()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save recording: {exc}")
            return

        def worker():
            try:
                wav_trimmed = trim_wav_silence(wav_path)
                embedding_vec = np.array(get_audio_embedding(wav_trimmed))

                # Delete audio artefacts
                try:
                    wav_trimmed.unlink(missing_ok=True)
                    if wav_trimmed != wav_path:
                        wav_path.unlink(missing_ok=True)
                except Exception:
                    pass

                users = self.session.query(User).all()
                if not users:
                    self.after(
                        0,
                        lambda: [
                            messagebox.showinfo("No data", "No registered users found."),
                            self._build_verify_form(),
                        ],
                    )
                    return

                embeddings_matrix = np.array([json.loads(u.embedding) for u in users])

                def cos_sim(a, b):
                    # Ensure both vectors are 1-D before computing the cosine
                    a_flat = np.ravel(a)
                    b_flat = np.ravel(b)
                    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
                    return float(np.dot(a_flat, b_flat) / denom) if denom else 0.0

                similarities = np.array([cos_sim(embedding_vec, e) for e in embeddings_matrix])

                threshold = float(self.config.get("similarity_threshold", 0.8))
                best_idx = int(np.argmax(similarities))
                best_score = similarities[best_idx]

                if best_score >= threshold:
                    user = users[best_idx]
                    msg = (
                        "Match found!\n"
                        f"Name: {user.full_name()}\n"
                        f"DOB: {user.date_of_birth.strftime('%Y-%m-%d')}\n"
                        f"Similarity: {best_score:.2f}"
                    )

                    self.after(
                        0,
                        lambda m=msg: [
                            messagebox.showinfo("Verified", m),
                            self._build_verify_form(),
                        ],
                    )
                else:
                    msg = (
                        "No matching user found.\n"
                        f"Best similarity: {best_score:.2f} (threshold {threshold})"
                    )
                    self.after(
                        0,
                        lambda m=msg: [
                            messagebox.showwarning("No Match", m),
                            self._build_verify_form(),
                        ],
                    )
            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # Admin Panel
    def _build_admin_panel(self):
        if hasattr(self, "_set_active_tab"):
            self._set_active_tab("Admin")

        # Reload config from disk in case it changed elsewhere
        self.config = load_config()

        self._clear_frame()

        # ------------------------------------------------------------
        # Layout – dedicate remaining vertical space to the user list
        # ------------------------------------------------------------

        admin_inner = Frame(self.content_frame, bg=self.content_frame.cget("bg"))
        admin_inner.pack(fill="both", expand=True)

        # Use grid to control row weights (last row stretches)
        admin_inner.columnconfigure(0, weight=1)
        admin_inner.rowconfigure(5, weight=1)

        # --- Title ----------------------------------------------------
        ttk.Label(
            admin_inner,
            text="Admin Panel",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        # --- Similarity threshold section -----------------------------
        self.threshold_var = StringVar(value=str(self.config.get("similarity_threshold", 0.8)))

        threshold_frame = Frame(admin_inner, bg=self.content_frame.cget("bg"))
        threshold_frame.grid(row=1, column=0, sticky="ew")
        threshold_frame.columnconfigure(1, weight=1)

        ttk.Label(threshold_frame, text="Similarity Threshold (0-1):").grid(row=0, column=0, sticky="w")
        ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=15).grid(
            row=0, column=1, sticky="w", pady=2, padx=(5, 0)
        )
        ttk.Button(
            threshold_frame, text="Save", command=self._save_settings, width=12
        ).grid(row=0, column=2, padx=10)

        # --- Script management section --------------------------------
        ttk.Separator(admin_inner, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=10)

        scripts_frame = Frame(admin_inner, bg=self.content_frame.cget("bg"))
        scripts_frame.grid(row=3, column=0, sticky="ew")
        scripts_frame.columnconfigure(1, weight=1)

        self.reg_script_var = StringVar(value=self.config.get("registration_script", ""))
        self.ver_script_var = StringVar(value=self.config.get("verification_script", ""))

        ttk.Label(scripts_frame, text="Registration Script:").grid(row=0, column=0, sticky="w")
        ttk.Entry(scripts_frame, textvariable=self.reg_script_var, width=60).grid(
            row=0, column=1, sticky="ew", pady=2
        )

        ttk.Label(scripts_frame, text="Verification Script:").grid(row=1, column=0, sticky="w")
        ttk.Entry(scripts_frame, textvariable=self.ver_script_var, width=60).grid(
            row=1, column=1, sticky="ew", pady=2
        )

        ttk.Button(
            scripts_frame, text="Save Scripts", command=self._save_scripts, width=15
        ).grid(row=2, column=0, columnspan=2, pady=10)

        # --- Users list ----------------------------------------------
        ttk.Separator(admin_inner, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=10)

        ttk.Label(
            admin_inner,
            text="Registered Users:",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=5, column=0, sticky="w")

        # Canvas + scrollbar hosted in its own container (row 6, weight=1)
        list_container = Frame(admin_inner, bg=self.content_frame.cget("bg"))
        list_container.grid(row=6, column=0, sticky="nsew")
        admin_inner.rowconfigure(6, weight=1)

        canvas = Canvas(
            list_container,
            borderwidth=0,
            background=self.content_frame.cget("bg"),
            highlightthickness=0,
        )
        v_scroll = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        rows_frame = Frame(canvas, bg=self.content_frame.cget("bg"))
        canvas.create_window((0, 0), window=rows_frame, anchor="nw")

        def _on_frame_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        rows_frame.bind("<Configure>", _on_frame_config)

        users = self.session.query(User).all()

        if not users:
            ttk.Label(rows_frame, text="No users registered yet.").pack(pady=5)
        else:
            for user in users:
                row = Frame(rows_frame, bg=self.content_frame.cget("bg"))
                row.pack(fill="x", pady=2)

                ttk.Label(row, text=f"{user.first_name} {user.last_name}").pack(side="left", padx=5)

                ttk.Button(
                    row,
                    text="Delete",
                    command=lambda uid=user.id, uname=user.full_name(): self._confirm_delete_user(uid, uname),
                    width=10,
                ).pack(side="right", padx=5)

        self.update_idletasks()

    def _save_settings(self):
        try:
            value = float(self.threshold_var.get())
            if not 0 <= value <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Threshold must be a number between 0 and 1.")
            return

        self.config["similarity_threshold"] = value
        save_config(self.config)
        messagebox.showinfo("Success", "Settings updated.")

    def _save_scripts(self):
        self.config["registration_script"] = self.reg_script_var.get()
        self.config["verification_script"] = self.ver_script_var.get()
        save_config(self.config)
        messagebox.showinfo("Success", "Scripts updated.")

    # ------------------------------------------------------------
    # User deletion helpers
    # ------------------------------------------------------------

    def _confirm_delete_user(self, user_id: int, user_name: str) -> None:
        """Ask for confirmation and delete the user if confirmed.

        Parameters
        ----------
        user_id:
            Primary key of the user in the *users.json* store.
        user_name:
            Display name shown in the confirmation dialogue.
        """

        if not messagebox.askyesno(
            "Confirm Deletion", f"Are you sure you want to delete {user_name}?"
        ):
            return

        if delete_user(user_id):
            messagebox.showinfo("Success", f"{user_name} has been deleted.")
        else:
            messagebox.showerror("Error", "User not found – nothing deleted.")

        # Refresh the admin panel to reflect changes immediately.
        self._build_admin_panel()


if __name__ == "__main__":
    app = VoiceAuthApp()
    app.mainloop()
