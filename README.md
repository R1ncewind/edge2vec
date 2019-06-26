# edge2evc

The open source code for our paper "edge2vec: Learning Node Representation Using Edge Semantics".

## Installation

Install in development mode with::

```bash
$ git clone https://github.com/RoyZhengGao/edge2vec.git
$ cd edge2vec
$ pip install -e .
```

## Usage

### Dataset

The dataset we offer for test is `data/data.csv`. The data contains four
columns, which refer to Source ID, Target ID, Edge Type, Edge ID. And
columns are separated by space ' '.

For unweighted graph, please see unweighted_graph.txt. The four columns
are Source ID, Target ID, Edge Type, Edge ID. And columns are separated
by space ' '. For weighted graph, please see weighted_graph.txt. The five
columns are Source ID, Target ID, Edge Type, Edge Weight, Edge ID. And
columns are separated by space ' '.

### Run the code

There are two steps for running the code. First, to calculate transition
matrix in heterogeneous networks, run `edge2vec-transition` from the shell:

```bash
$ edge2vec-transition \
    --input data/data.csv \
    --output data/matrix.txt \
    --type_size 3 \
    --em_iteration 5 \
    --walk-length 3
```

The output is `matrix.txt`, which stores the edge transition matrix.
Second, run `edge2vec` to the node embeddings via biased random walk.
To use it from the shell:

```bash
$ edge2vec \
    --input data/data.csv \
    --matrix data/matrix.txt \
    --output data/vector.txt \
    --dimensions 128 \
    --walk-length 3 \
    --p 1 \
    --q 1
```

The output is the node embedding file `vector.txt`.

Data repository for medical dataset in the link: http://ella.ils.indiana.edu/~gao27/data_repo/edge2vec%20vector.zip
or https://figshare.com/articles/edge2vec_vector_zip/8097539 (It is a
re-computed version so the evaluation output may be a little bit different
with the paper reported results.)

## Citations

if you use the code, please cite:

- Gao, Zheng, Gang Fu, Chunping Ouyang, Satoshi Tsutsui, Xiaozhong Liu, and Ying Ding. "edge2vec: Learning Node Representation Using Edge Semantics." arXiv preprint arXiv:1809.02269 (2018).

## License

The code is released under BSD 3-Clause License. 

## Contributor

* **Zheng Gao** - [gao27@indiana.edu](gao27@indiana.edu) <br />
