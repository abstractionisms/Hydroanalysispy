# Hydrology Analysis Project

This repository contains scripts and data for a hydrology analysis project. The project focuses on analyzing streamflow data and its relationship with climate data.

**Please note:** This project is being worked on in my spare time, so development may be intermittent.

## Project Description

The project includes the following components:

* **Data Screening:** Scripts to assess the availability of climate data for specific hydrological sites.
* **Trend Analysis:** Scripts to analyze trends in streamflow data (annual and monthly).
* **Climate Correlation:** Scripts to explore the correlation between streamflow and climate data (temperature and precipitation).
* **NWM Evaluation:** Scripts (in development) to evaluate streamflow data against the National Water Model (NWM).
* **Watershed Delineation:** Scripts (in development) to delineate watershed boundaries.
* **Dashboard:** A Streamlit dashboard (in development) for visualizing streamflow data.

## Data

The repository includes the following data files:

* `nwis_inventory_with_latlon.txt`:  Inventory of streamflow sites with location data.
* `config.json`:  Configuration file containing settings for the analysis.

## Scripts

The following Python scripts are included:

* `climate_corr.py`:  Script for climate correlation analysis.
* `q_trends.py`: Script for streamflow trend analysis.
* `screen_climate_availability.py`: Script for climate data availability screening.
* `project2_climate_corr.py`:  An older version of the climate correlation script.
* `project3_nwm_eval.py`:  Script for NWM evaluation (under development).
* `project4_watershed.py`: Script for watershed delineation (under development).
* `project5_dashboard.py`:  Streamlit dashboard application (under development).

## Log Files

The following log files are included:

* `climate_corr.log`: Log file for the climate correlation script.
* `trend_analysis.log`: Log file for the trend analysis script.
* `climate_screening.log`: Log file for the climate screening script.

## Getting Started

To use this project, you'll need the following:

* Python 3.x
* Required Python libraries (install using `pip install -r requirements.txt`)
* (Optional) AWS credentials configured if you intend to run the NWM evaluation scripts.

1.  Clone the repository:  
    ```bash
    git clone (https://github.com/abstractionisms/Hydroanalysispy)
    cd Hydrology-Analysis  # Or whatever you wanna name it
    ```
2.  (Optional) Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate      # On Windows (personally I am using conda...)
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Modify the `config.json` file to match your desired settings.
5.  Run the scripts from the command line (e.g., `python q_trends.py`).

## Contributing

Contributions to this project are welcome!  Since I work on this in my spare time, responses to issues and pull requests may be delayed.
