# QWOP Reinforcement Learning Project

A reinforcement learning project for training an agent to play QWOP using qwop-gym and Stable Baselines3.

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to the QWOP directory
cd "c:\Sidd all in one\CSE is Ez\FAI\QWOP"

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell):
.\.venv\Scripts\Activate.ps1

# Or Command Prompt:
.\.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup qwop-gym

The project uses [qwop-gym](https://github.com/smanolloff/qwop-gym) environment. After installation, patch the QWOP game:

```bash
# Download and patch QWOP.min.js
(Invoke-WebRequest https://www.foddy.net/QWOP.min.js).Content | qwop-gym patch
```

### 4. Configure Browser Paths

Update the browser and ChromeDriver paths in your code:
- **Browser**: Path to Chrome/Brave executable (e.g., `C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe`)
- **Driver**: Path to ChromeDriver executable (e.g., `C:\Program Files\chromedriver-win64\chromedriver-win64\chromedriver.exe`)

Download ChromeDriver from: https://sites.google.com/chromium.org/driver/

## Project Structure

```
QWOP/
├── My-RL/
│   ├── playground.ipynb    # Interactive testing and experimentation
│   ├── main.py             # Main training script (to be implemented)
│   ├── QWOP.min.js         # Patched QWOP game file
│   └── game/               # Game assets and environment files
├── QWOP-RL-Reference/      # Reference implementation
├── requirements.txt        # Python dependencies
├── .venv/                  # Virtual environment
└── README.md              # This file
```

## Usage

### Interactive Testing (Jupyter Notebook)

Open `My-RL/playground.ipynb` to:
- Test the QWOP environment
- Run random actions
- Visualize game statistics
- Experiment with different configurations

### Training (To be implemented)

```bash
cd My-RL
python main.py
```

## Key Dependencies

- **qwop-gym**: Gymnasium environment for QWOP game
- **gymnasium**: OpenAI's toolkit for RL environments
- **stable-baselines3**: High-quality RL algorithm implementations
- **numpy**: Numerical computing
- **tensorflow**: Deep learning framework
- **selenium**: Web browser automation
- **pynput**: Keyboard control

## QWOP Environment

The qwop-gym environment provides:
- **Action Space**: 16 discrete actions (key combinations of Q, W, O, P)
  - 0: none, 1: Q, 2: W, 3: O, 4: P
  - 5: Q+W, 6: Q+O, 7: Q+P, 8: W+O, 9: W+P, 10: O+P
  - 11-15: 3 and 4-key combinations
- **Observation Space**: 60-dimensional normalized state vector (body part positions, angles, velocities)
- **Rewards**: Based on forward progress, with penalties for time and falling

### Environment Options
- `stat_in_browser=True`: Display statistics in browser
- `auto_draw=True`: Automatically render each frame
- `reduced_action_set=True`: Use only 9 non-redundant actions

## Development

To deactivate the virtual environment:

```bash
deactivate
```

## Reference

This project is based on:
- **qwop-gym**: https://github.com/smanolloff/qwop-gym
- **QWOP-RL Reference**: Included in `QWOP-RL-Reference/` directory

## Notes

- Python 3.8+ required
- Ensure virtual environment is activated when working
- The browser window will open automatically when creating the environment
- Use Ctrl+C to stop training/testing
