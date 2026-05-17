"""
GUI Application for Hate Speech Detection
Simple Tkinter interface for real-time predictions
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
from rnn_model import BiLSTMHateSpeechClassifier
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences
import os


class HateSpeechDetectorGUI:
    """GUI for hate speech detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Filipino-English Hate Speech Detector")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Model components (will be loaded)
        self.model = None
        self.vocab = None
        self.preprocessor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create GUI components
        self.create_widgets()
        
        # Try to load model automatically
        self.load_model_auto()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            title_frame,
            text="🛡️ Bilingual Hate Speech Detector",
            font=("Arial", 20, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Filipino & English Text Classification",
            font=("Arial", 10)
        )
        subtitle_label.pack()
        
        # Model status
        status_frame = ttk.Frame(self.root, padding="5")
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Model Status: Not Loaded",
            font=("Arial", 9),
            foreground="red"
        )
        self.status_label.pack()
        
        # Input section
        input_frame = ttk.LabelFrame(self.root, text="Enter Text to Analyze", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            font=("Arial", 11)
        )
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        placeholder = "Type or paste text here (Filipino or English)...\nExample: 'Salamat sa suporta!' or 'This is a test message.'"
        self.text_input.insert("1.0", placeholder)
        self.text_input.config(fg='grey')
        
        # Bind focus events for placeholder
        self.text_input.bind("<FocusIn>", self.on_entry_click)
        self.text_input.bind("<FocusOut>", self.on_focusout)
        
        # Buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        
        self.analyze_button = ttk.Button(
            button_frame,
            text="🔍 Analyze Text",
            command=self.analyze_text,
            style="Accent.TButton"
        )
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_text
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = ttk.Button(
            button_frame,
            text="📁 Load Model",
            command=self.load_model_manual
        )
        self.load_button.pack(side=tk.RIGHT, padx=5)
        
        # Results section
        results_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Result display
        self.result_label = ttk.Label(
            results_frame,
            text="Prediction: -",
            font=("Arial", 16, "bold")
        )
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(
            results_frame,
            text="Confidence: -",
            font=("Arial", 12)
        )
        self.confidence_label.pack(pady=5)
        
        # Progress bar (for loading indication)
        self.progress = ttk.Progressbar(
            results_frame,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=10)
        self.progress.pack_forget()  # Hide initially
    
    def on_entry_click(self, event):
        """Handle click on text input (remove placeholder)"""
        if self.text_input.get("1.0", tk.END).strip().startswith("Type or paste"):
            self.text_input.delete("1.0", tk.END)
            self.text_input.config(fg='black')
    
    def on_focusout(self, event):
        """Handle focus out (restore placeholder if empty)"""
        if not self.text_input.get("1.0", tk.END).strip():
            placeholder = "Type or paste text here (Filipino or English)...\nExample: 'Salamat sa suporta!' or 'This is a test message.'"
            self.text_input.insert("1.0", placeholder)
            self.text_input.config(fg='grey')
    
    def load_model_auto(self):
        """Try to load model automatically"""
        try:
            if not os.path.exists('best_bilstm_model.pt') or not os.path.exists('vocabulary.pkl'):
                self.status_label.config(
                    text="Model Status: Model files not found. Please train the model first.",
                    foreground="orange"
                )
                return
            
            self.load_model()
        except Exception as e:
            print(f"Auto-load failed: {e}")
    
    def load_model_manual(self):
        """Load model manually (button click)"""
        try:
            self.progress.pack()
            self.progress.start()
            self.root.update()
            
            self.load_model()
            
            self.progress.stop()
            self.progress.pack_forget()
            
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
    
    def load_model(self):
        """Load trained model and vocabulary"""
        # Load vocabulary
        self.vocab = Vocabulary.load('vocabulary.pkl')
        
        # Load preprocessor
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=False,
            remove_punctuation=False
        )
        
        # Initialize model
        self.model = BiLSTMHateSpeechClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            pad_idx=self.vocab.word2idx[self.vocab.PAD_TOKEN]
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load('best_bilstm_model.pt', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.status_label.config(
            text=f"Model Status: Loaded ✓ (Device: {self.device})",
            foreground="green"
        )
    
    def clear_text(self):
        """Clear input and results"""
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="Prediction: -", foreground="black")
        self.confidence_label.config(text="Confidence: -")
    
    def analyze_text(self):
        """Analyze the input text"""
        if self.model is None:
            messagebox.showwarning(
                "Model Not Loaded",
                "Please load the model first by clicking 'Load Model' button."
            )
            return
        
        # Get text
        text = self.text_input.get("1.0", tk.END).strip()
        
        if not text or text.startswith("Type or paste"):
            messagebox.showwarning("Empty Input", "Please enter some text to analyze.")
            return
        
        try:
            # Show progress
            self.progress.pack()
            self.progress.start()
            self.root.update()
            
            # Preprocess
            processed_text = self.preprocessor.preprocess(text)
            
            # Convert to sequence
            sequence = self.vocab.text_to_sequence(processed_text, max_length=100)
            padded_seq, length = pad_sequences([sequence], max_length=100)
            
            # Convert to tensors
            text_tensor = torch.LongTensor(padded_seq).to(self.device)
            length_tensor = torch.LongTensor(length).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = model(text_tensor, length_tensor)
                probability = torch.sigmoid(output).item()
            
            # Interpret results with adjusted threshold (0.8 for better accuracy)
            threshold = 0.8
            is_hate = probability > threshold
            confidence = probability if is_hate else (1 - probability)
            
            # Update UI
            if is_hate:
                prediction_text = "⚠️ HATE SPEECH DETECTED"
                color = "red"
            else:
                prediction_text = "✅ NO HATE SPEECH"
                color = "green"
            
            self.result_label.config(text=f"Prediction: {prediction_text}", foreground=color)
            self.confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")
            
            # Hide progress
            self.progress.stop()
            self.progress.pack_forget()
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = HateSpeechDetectorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
