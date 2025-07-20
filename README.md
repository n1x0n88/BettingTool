# BettingTool

An end-to-end sports analytics pipeline for feature engineering, model training, and live API/dashboard deployment.

---

## Setup

Before you begin, ensure you have:

- Python 3.11  
- Docker Desktop (for API & dashboard)  
- Homebrew (macOS) for easy tooling installs  

Clone the repo and install dependencies:

```bash
git clone https://github.com/n1x0n88/BettingTool.git
cd BettingTool

# create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# install Python packages
pip install -r requirements.txt
