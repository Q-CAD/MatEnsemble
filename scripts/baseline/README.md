# MatEnsemble Installation Guide (CADES-baseline)

### Prerequisites
- Access to CADES computing environment
- Conda package manager
- Sufficient storage space for dependencies

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Q-CAD/MatEnsemble.git
cd Matensemble/scripts/baseline
```

2. Make the installation script executable:
```bash
chmod +x install_baseline.sh
```

3. Run the installation with your desired environment path:
```bash
./install_baseline.sh /path/to/your/conda/environment
```

For example, to install in your home directory:
```bash
./install_baseline.sh ~/envs/matensemble_env
```

### Important Notes

- Choose a location with sufficient storage space as the installation builds dependencies from scratch
- If the specified environment directory already exists, you will be prompted:
  - Enter 'y' to remove the existing environment and continue
  - Enter 'n' to cancel the installation
- For long installation processes, consider using `nohup`:
```bash
nohup ./install_baseline.sh /path/to/your/conda/environment > install.log 2>&1 &
```

### Post-Installation

Activate your environment using:
```bash
conda activate /path/to/your/conda/environment
```

### Troubleshooting

If you encounter issues:
- Check available disk space: `df -h`
- Verify conda is accessible: `conda --version`
- Review installation logs if using nohup: `tail -f install.log`