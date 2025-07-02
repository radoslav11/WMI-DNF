# Approximate weighted model integration on DNF structures

This repository contains a simplistic implementation of the Weighted Model Integration algorithm on DNF Structures, which was proposed in the KR 2020 [paper](https://arxiv.org/abs/2002.06726) and it's extended [journal version](https://doi.org/10.1016/j.artint.2022.103753). *Some of the decision choices in the implementation are suboptimal, so it is advised to simply use the code as a reference.*  

## External requirements

Part of the algorithm is that we should initially find the "volumes" of the polytopes independently. For this will need to install [Latte](https://www.math.ucdavis.edu/~latte/) into the latte-distro folder (top level). Please refer to the driver (implemented in `utils/runLatte.py`) for more details on how we implement the interaction with Latte, and in particular what we pass to it.

The implementation also requires `tqdm` for progress bars:
```bash
pip install tqdm
```

## Setting up an experiment

You should refer to `main.py` for setting up an experiment. We run Latte by giving a weighted function in the form of a list of monomials. Refer to `utils/weightFunction.py` for the exact details. 

## Examples

You can find examples in the `examples` folder. It contains a dedicated README file with more details.

## Additional parameters

Due to weighted sampling from a polytope being non-trivial, we use a hit-and-run approach. More precisely, we create a "hidden" dimension which essentially is the projected weight, and then we uniformly sample from this new shape. Even though finding the closed form for the distance until when we "hit" the wall of this new figure is non-trivial, we can empirically find it by binary searching. An alternative approach that could be done is to find this distance by solving an optimization problem.

We regulate the constant for the number of hit and run steps inside of `utils/polytopeSampling.py`.

##  Citing this paper
If you make use of this code, or its accompanying paper, 
please cite this work as follows:

```
@article{abboud2022approximate,
  title={Approximate weighted model integration on DNF structures},
  author={Ralph Abboud and 
         {\.I}smail {\.I}lkan Ceylan and 
         Radoslav Dimitrov},
  journal={Artificial Intelligence},
  volume={311},
  pages={103753},
  year={2022},
  publisher={Elsevier}
}
```
