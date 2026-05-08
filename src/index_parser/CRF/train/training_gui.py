import tkinter as tk
from tkinter import filedialog, messagebox
from train_crf import Train  # Importing the Train class for model training

class TrainModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Train Model")  # Set the title of the window
        self.root.geometry("500x400")  # Set the size of the window
        self.root.configure(bg='#1E64C8')  # Set background color

        # Path entry and browse button
        self.label = tk.Label(root, text="Select CSV File:", bg='#1E64C8', fg='white')  # Label for file path
        self.label.grid(row=0, column=0, padx=10, pady=10, sticky='e')  # Positioning the label in the grid

        self.entry = tk.Entry(root, width=40)  # Entry widget for the file path
        self.entry.grid(row=0, column=1, padx=10, pady=10)  # Positioning the entry widget

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_path, bg='white')  # Browse button
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)  # Positioning the browse button

        # Model name entry
        self.model_name_label = tk.Label(root, text="Model Name:", bg='#1E64C8', fg='white')  # Label for model name
        self.model_name_label.grid(row=1, column=0, padx=10, pady=5, sticky='e')  # Positioning the model name label
        self.model_name_entry = tk.Entry(root)  # Entry widget for model name
        self.model_name_entry.grid(row=1, column=1, padx=10, pady=5)  # Positioning the model name entry widget

        # c1, c2, and max_iterations entries (hyperparameters for training)
        self.c1_label = tk.Label(root, text="c1=", bg='#1E64C8', fg='white')  # Label for c1 hyperparameter
        self.c1_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')  # Positioning the c1 label
        self.c1_entry = tk.Entry(root)  # Entry widget for c1
        self.c1_entry.insert(0, "0.01")  # Default value for c1
        self.c1_entry.grid(row=2, column=1, padx=10, pady=5)  # Positioning the c1 entry widget

        self.c2_label = tk.Label(root, text="c2=", bg='#1E64C8', fg='white')  # Label for c2 hyperparameter
        self.c2_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')  # Positioning the c2 label
        self.c2_entry = tk.Entry(root)  # Entry widget for c2
        self.c2_entry.insert(0, "0.01")  # Default value for c2
        self.c2_entry.grid(row=3, column=1, padx=10, pady=5)  # Positioning the c2 entry widget

        self.max_iterations_label = tk.Label(root, text="max_iterations=", bg='#1E64C8', fg='white')  # Label for max_iterations
        self.max_iterations_label.grid(row=4, column=0, padx=10, pady=5, sticky='e')  # Positioning the max_iterations label
        self.max_iterations_entry = tk.Entry(root)  # Entry widget for max_iterations
        self.max_iterations_entry.insert(0, "1000")  # Default value for max_iterations
        self.max_iterations_entry.grid(row=4, column=1, padx=10, pady=5)  # Positioning the max_iterations entry widget

        # Train button
        self.train_button = tk.Button(root, text="Train", command=self.on_train_button_click, bg='white', width=25)  # Train button
        self.train_button.grid(row=8, column=1, pady=20)  # Positioning the train button

    def browse_path(self):
        # Open a file dialog to select a CSV file
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select a CSV File", filetypes=filetypes)  # File dialog to select file
        if path:
            self.entry.delete(0, tk.END)  # Clear the entry field
            self.entry.insert(0, path)    # Insert the selected path into the entry field

    def on_train_button_click(self):
        trainer = Train()  # Create an instance of the Train class
        path = self.entry.get()  # Get the file path from the entry widget
        if path:
            model_name = self.model_name_entry.get()  # Get the model name
            c1_value = self.c1_entry.get()  # Get the c1 value
            c2_value = self.c2_entry.get()  # Get the c2 value
            max_iterations_value = self.max_iterations_entry.get()  # Get the max_iterations value
            accuracy = trainer.train(path, model_name, float(c1_value), float(c2_value), int(max_iterations_value))  # Train the model
            # Show a message box with the training result
            messagebox.showinfo("Training", f"Training completed for model '{model_name}' with path: {path}\n"
                                            f"c1={c1_value}, c2={c2_value}, max_iterations={max_iterations_value}, accuracy={accuracy}")
        else:
            # Show a warning if no valid file path is selected
            messagebox.showwarning("Input Error", "Please select a valid CSV file.")

# Function to run the GUI
def run_gui():
    root = tk.Tk()  # Create the main window
    gui = TrainModelGUI(root)  # Create an instance of the TrainModelGUI class
    root.mainloop()  # Start the Tkinter event loop

# Start the GUI when the script is run directly
if __name__ == "__main__":
    run_gui()
