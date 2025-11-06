# How to Run the Unified AI System

This document provides instructions on how to set up and run the Unified AI System.

## 1. Project Setup

The project can be set up by running the `automated_integration_system.py` script, which generates the complete directory structure and necessary files. After that, run the `setup.sh` script to install dependencies and prepare the environment.

```bash
python3 automated_integration_system.py
cd unified_ai_system
./scripts/setup.sh
```

## 2. Running the Main Application

The main application can be run by executing the `main.py` script from within the `unified_ai_system` directory. Make sure to activate the virtual environment first.

```bash
cd unified_ai_system
source venv/bin/activate
python3 main.py
```

## 3. Running the Demo Scripts

The project includes two demo scripts that showcase the system's functionality:

- `complete_demo_e2e.py`: An end-to-end demonstration of the system.
- `complete_system_demo.py`: A demonstration of the system with multiple agents.

To run the demos, execute the scripts from the root directory:

```bash
python3 complete_demo_e2e.py
python3 complete_system_demo.py
```

## 4. Running Tests

The test suite can be run from within the `unified_ai_system` directory. Make sure to activate the virtual environment first.

```bash
cd unified_ai_system
source venv/bin/activate
pytest tests/
```
