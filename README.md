# Gray Swan Tropical Cyclone Forecasting with AI?

This repository contains code for the paper: Can AI Weather Models Predict Out-of-Distribution Gray Swan Tropical Cyclones? by Y. Q. Sun, P. Hassanzadeh, M. Zand, A. Chattopadhyay, J. Weare, and D. Abbot [paper](https://arxiv.org/pdf/2410.14932). 


We use the original FourCastNet with modifications for our customized training sets. Please refer to their repository for the foundational model details and other information (https://github.com/NVlabs/FourCastNet).

The necessary data to reproduce the results, including the weights of all trained models and indices of dates that are removed in each training dataset, can be found on Zenodo at (https://zenodo.org/uploads/13835657) and (https://zenodo.org/uploads/13834149).


## Citation
If you use this code, please cite the following work:
```
@article{sun2024aiweathermodelspredict,
      title={Can AI weather models predict out-of-distribution gray swan tropical cyclones?}, 
      author={Y. Qiang Sun and Pedram Hassanzadeh and Mohsen Zand and Ashesh Chattopadhyay and Jonathan Weare and Dorian S. Abbot},
      year={2024},
      eprint={2410.14932},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph},
      url={https://arxiv.org/abs/2410.14932}, 
}
```