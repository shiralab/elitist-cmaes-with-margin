# (1+1)-CMA-ES with margin

The (1+1)-CMA-ES with margin is an extension of CMA-ES with margin [1] with the elite-preserving strategy. The (1+1)-CMA-ES with margin works effectively on binary and integer black-box optimization problems as well as mixed-integer black-box optimization problems.

## Sample Codes
- `sample_script_mixed.ipynb`
    - Sample code for mixed-integer black-box optimization problems
- `sample_script_integer.ipynb`
    - Sample code for 
    integer black-box optimization problems
- `sample_script_binary.ipynb`
    - Sample code for 
    binary black-box optimization problems
- `demonstration_of_mean_vector_discretization.ipynb`
    - Experiment about mean vector discretization (see Sec.3.2 and Fig.1 in our paper for GECCO 2023)

## Environment
We tested this source code with the following environment.
```
numpy==1.24.3
scipy==1.10.1
```

## Reference Information
Yohei Watanabe, Kento Uchida, Ryoki Hamano, Shota Saito, Masahiro Nomura, and Shinichi Shirakawa: (1+1)-CMA-ES with Margin for Discrete and Mixed-Integer Problems, Genetic and Evolutionary Computation Conference Companion (GECCO 2023), Lisbon, Portugal (hybrid), July 15-19, 2023. [[arXiv](https://arxiv.org/abs/2305.00849)]

## Bibliography
[1] Ryoki Hamano, Shota Saito, Masahiro Nomura, and Shinichi Shirakawa, CMA-ES with Margin: Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization, In Genetic and Evolutionary Computation Conference (GECCO ’22), July 9–13, 2022, Boston, MA, USA. ACM, New York, NY, USA, 9 pages. [https://doi.org/10.1145/3512290.3528827](https://doi.org/10.1145/3512290.3528827) [[arXiv](https://arxiv.org/abs/2205.13482)]

## Acknowledgement
Our code is based on the source code of CMA-ES with margin [[link](https://github.com/EvoConJP/CMA-ES_with_Margin)]