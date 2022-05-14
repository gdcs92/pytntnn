# pytntnn
A Python implementation of the TNT-NN method for Non-negative Least Squares problems.

Derived from Erich Frahm and Joseph Myre's Matlab implementation available at https://github.com/ProfMyre/tnt-nn.

## Installation
Just copy and paste the subdirectory `pytntnn` (not the repository root!) to where the script from which it will be imported resides.
The main function can then be imported as `from pytntnn import tntnn`. See [tntnn_x_sklearn_simple_comparison.py](https://github.com/gdcs92/pytntnn/blob/master/tntnn_x_sklearn_simple_comparison.py) for an example.

## License
This work is released under the [GNU GPL v3.0 License](https://github.com/gdcs92/pytntnn/blob/develop/LICENSE).

## Citation
If you use this software, please cite the papers by the original authors of the TNT-NN method:

**TNT-NN: A Fast Active Set Method for Solving Large Non-Negative Least Squares Problems**
```bibtex
@article{myre2017tnt,
title={TNT-NN: A Fast Active Set Method for Solving Large Non-Negative Least Squares Problems},
author={Myre, Joseph M and Frahm, E and Lilja, David J and Saar, Martin O},
journal={Procedia Computer Science},
volume={108},
pages={755--764},
year={2017},
publisher={Elsevier}
}
```

**TNT: A Solver for Large Dense Least-Squares Problems that Takes Conjugate Gradient from Bad in Theory, to Good in Practice**
```bibtex
@inproceedings{myre2018tnt,
title={TNT: A Solver for Large Dense Least-Squares Problems that Takes Conjugate Gradient from Bad in Theory, to Good in Practice},
author={Myre, Joseph M and Frahm, Erich and Lilja, David J and Saar, Martin O},
booktitle={2018 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
pages={987--995},
year={2018},
organization={IEEE}
}
```
