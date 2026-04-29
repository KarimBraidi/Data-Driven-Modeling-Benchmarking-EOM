## Repository Structure

```
Data Driven Modeling Project/
│
├── hoop data/
│   ├── HoopSINDy_Pipeline.ipynb       # SINDy pipeline for the hoop system
│   ├── HoopPySR_Pipeline.ipynb        # PySR pipeline for the hoop system
│   ├── OR_*.csv                       # Raw experimental data files
│   ├── hoop_full_sweep_results.csv    # Full hyperparameter sweep results
│   ├── hoop_pysr_sweep_results.csv    # PySR sweep results
│   ├── model_comparison_results.csv   # Model comparison summary
│   ├── sindy_analysis_results.csv     # SINDy analysis output
│   ├── run_sindy_analysis.py          # Script to run SINDy analysis
│   ├── _gen_pipeline.py               # Pipeline generation utility
│   ├── plot_*.html                    # Interactive result plots
│   └── *.png                         # Summary comparison plots
│
├── mass imbalance data/
│   ├── mass imbalance data/           # Raw sensor data files
│   └── pipeline/
│       ├── SINDy_Pipeline.ipynb       # SINDy pipeline for mass imbalance system
│       ├── symbolic.ipynb             # Symbolic regression notebook
│       ├── sindy.py                   # Core SINDy implementation
│       ├── library.py                 # Feature library construction
│       ├── preprocessing.py           # Data preprocessing utilities
│       ├── evaluation.py              # Model evaluation metrics
│       ├── visualization.py           # Plotting utilities
│       ├── experiments.py             # Experiment runner
│       ├── full_sweep_results.csv     # Sweep results
│       └── mega_sweep_results*.csv    # Extended sweep results
│
├── sindy_results/
│   ├── all_summaries.json             # Aggregated results across all runs
│   ├── all_threshold_grades.csv       # Threshold grading summary
│   ├── eom_comparison_report.csv      # EOM term comparison report
│   ├── bouncing_ball_1/               # Results for bouncing ball case 1
│   ├── bouncing_ball_2/               # Results for bouncing ball case 2
│   └── bouncing_ball_3/               # Results for bouncing ball case 3
│
├── ppt_plots/                         # Plots used in the presentation
├── hoop_ppt_plots/                    # Hoop-specific presentation plots
├── Data Driven Project.pptx           # Project presentation
└── Data Driven(1).pptx                # Updated presentation version
```

## How to Run

### Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn pysindy pysr jupyter
```

### Hoop System

1. Open `hoop data/HoopSINDy_Pipeline.ipynb` and run all cells to perform SINDy-based equation discovery on the hoop data.
2. Open `hoop data/HoopPySR_Pipeline.ipynb` and run all cells to run the PySR symbolic regression pipeline.
3. To re-run the full hyperparameter sweep from the command line:
   ```bash
   python "hoop data/run_sindy_analysis.py"
   ```

### Mass Imbalance System

1. Open `mass imbalance data/pipeline/SINDy_Pipeline.ipynb` and run all cells for the SINDy pipeline.
2. Open `mass imbalance data/pipeline/symbolic.ipynb` for symbolic regression experiments.

### Outputs

- Results are saved as CSV files in each pipeline folder.
- Interactive HTML plots are generated in `hoop data/`.
- Aggregated summaries are stored in `sindy_results/`.
