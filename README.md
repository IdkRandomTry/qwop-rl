# QWOP Project

A reinforcement learning project for training an agent to play QWOP.

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to the QWOP directory
cd "c:\Sidd all in one\CSE is Ez\FAI\QWOP"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python main.py
```

## Project Structure

```
QWOP/
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── venv/               # Virtual environment (created during setup)
└── README.md           # This file
```

## Dependencies

- **Gymnasium**: OpenAI's toolkit for developing RL algorithms
- **NumPy**: Numerical computing library
- **PyTorch**: Deep learning framework
- **Matplotlib**: Plotting and visualization
- **Pygame**: Game development library
- **tqdm**: Progress bar utility

## Development

To deactivate the virtual environment when done:

```bash
deactivate
```

## Notes

- Make sure Python 3.8+ is installed on your system
- The virtual environment should be activated whenever working on this project
- Dependencies are listed in `requirements.txt` for easy installation
