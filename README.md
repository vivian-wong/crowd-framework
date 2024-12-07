# Crowd framework paper code
Paper name TBD

## Installation

```bash
git clone https://github.com/vivian-wong/crowd-framework
cd crowd-framework
# create conda virtual environment
conda create --name crowd-framework python=3.10 
conda activate crowd-framework
# install prerequisites
pip install -r requirements.txt
# install pytorch geometric and pytorch geometric temporal
python install_pyg.py
```

## Usage


## Examples
Check the examples/ directory for simplified demo notebooks.

## Reproducing paper experiments 
To run all experiments as detailed in the paper, run 
```
bash reproduce_paper_experiments.sh
```
and generate plots with the jupyter notebook experiments/plot_results.ipynb

## Contributing
Contributions are welcome! Please read the CONTRIBUTING.md for guidelines.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/vivian-wong/crowd-framework/blob/master/LICENSE) file for details.
