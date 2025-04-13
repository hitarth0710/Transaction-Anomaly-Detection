import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Add the project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import modeling

class AnomalyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transaction Anomaly Detector")
        self.root.geometry("700x500")
        self.root.configure(padx=20, pady=20)
        
        # Load model
        try:
            self.model = modeling.load_model('models/isolation_forest_model.pkl')
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create and configure title
        title_style = ttk.Style()
        title_style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        title_label = ttk.Label(main_frame, text="Transaction Anomaly Detection", style="Title.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Input Frame
        input_frame = ttk.LabelFrame(main_frame, text="Transaction Details")
        input_frame.pack(fill=tk.X, pady=10)
        
        # Transaction Amount
        ttk.Label(input_frame, text="Transaction Amount ($):").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.transaction_amount_var = tk.DoubleVar(value=100.0)
        transaction_entry = ttk.Entry(input_frame, textvariable=self.transaction_amount_var, width=20)
        transaction_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Average Transaction Amount
        ttk.Label(input_frame, text="Average Transaction Amount ($):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.avg_amount_var = tk.DoubleVar(value=90.0)
        avg_entry = ttk.Entry(input_frame, textvariable=self.avg_amount_var, width=20)
        avg_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Frequency of Transactions
        ttk.Label(input_frame, text="Frequency of Transactions (per month):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.frequency_var = tk.DoubleVar(value=5.0)
        frequency_entry = ttk.Entry(input_frame, textvariable=self.frequency_var, width=20)
        frequency_entry.grid(row=2, column=1, padx=10, pady=5)
        
        # Quick samples frame
        samples_frame = ttk.LabelFrame(main_frame, text="Quick Samples")
        samples_frame.pack(fill=tk.X, pady=10)
        
        # Normal transaction button
        normal_btn = ttk.Button(samples_frame, text="Normal Transaction", 
                               command=self.load_normal_sample)
        normal_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Anomaly transaction button
        anomaly_btn = ttk.Button(samples_frame, text="Anomalous Transaction", 
                               command=self.load_anomaly_sample)
        anomaly_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Detect button
        detect_btn = ttk.Button(main_frame, text="Detect Anomaly", command=self.detect_anomaly)
        detect_btn.pack(pady=10)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(main_frame, text="Detection Result")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Result label
        self.result_var = tk.StringVar(value="Enter transaction details and click 'Detect Anomaly'")
        result_style = ttk.Style()
        result_style.configure("Result.TLabel", font=("Arial", 12))
        self.result_label = ttk.Label(self.result_frame, textvariable=self.result_var, 
                                    style="Result.TLabel", wraplength=600)
        self.result_label.pack(pady=20)
        
        # Visualization
        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Model loaded successfully" if self.model_loaded else "Model not loaded")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_normal_sample(self):
        """Load a sample normal transaction"""
        self.transaction_amount_var.set(95.0)
        self.avg_amount_var.set(100.0)
        self.frequency_var.set(6.0)
    
    def load_anomaly_sample(self):
        """Load a sample anomalous transaction"""
        self.transaction_amount_var.set(450.0)
        self.avg_amount_var.set(100.0)
        self.frequency_var.set(15.0)
    
    def detect_anomaly(self):
        """Detect if the transaction is an anomaly"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        try:
            # Get transaction details from input fields
            transaction = {
                'Transaction_Amount': self.transaction_amount_var.get(),
                'Average_Transaction_Amount': self.avg_amount_var.get(),
                'Frequency_of_Transactions': self.frequency_var.get()
            }
            
            # Check if transaction is an anomaly
            is_anomaly = modeling.predict_anomaly(self.model, transaction)
            
            # Update result display
            if is_anomaly:
                result_text = "ANOMALY DETECTED: This transaction is flagged as suspicious."
                self.result_var.set(result_text)
                self.update_visualization(transaction, is_anomaly=True)
            else:
                result_text = "No anomaly detected: This transaction appears to be normal."
                self.result_var.set(result_text)
                self.update_visualization(transaction, is_anomaly=False)
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def update_visualization(self, transaction, is_anomaly):
        """Update the visualization based on detection result"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create a simple visualization showing where this transaction falls
        x = np.arange(50, 500, 10)
        y = 100 * np.ones_like(x)  # baseline avg amount
        
        # Plot the baseline
        ax.scatter(x, y, alpha=0.1, color='blue', label='Normal Range')
        
        # Plot the threshold
        threshold = 250  # simplified threshold
        ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold')
        
        # Plot the current transaction
        marker_color = 'red' if is_anomaly else 'green'
        ax.scatter(transaction['Transaction_Amount'], transaction['Average_Transaction_Amount'], 
                 color=marker_color, s=100, label='Current Transaction')
        
        ax.set_xlabel('Transaction Amount')
        ax.set_ylabel('Average Transaction Amount')
        ax.set_title('Transaction Analysis')
        ax.legend()
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    root.mainloop()