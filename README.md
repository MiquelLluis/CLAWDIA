<span style="font-family:Times; font-size:4em; text-align:right; display:block;">CLAWDIA</span>

<span style="font-size:2em; text-align:right; display:block;">_**Cla**ssification of **W**aves via **Di**ctionary-based **A**lgorithms_</span>

<br>







# Introduction

**CLAWDIA** is a modular pipeline developed with the aim to facilitate data analysis tasks related to Gravitational Waves (GW) using Spars Dictionary Learning (SDL) techniques, with a particular focus in denoising and classification. Its modular design is intended to allow CLAWDIA to be used both as a whole classification pipeline and as a compillation of standalone routines and functions, making it a versatile tool for a wide range of use cases and applications.

The workflow is divided into two main stages: denoising and classification. In the denoising stage, the input signals are processed to reduce noise artifacts while striving to preserve the key features of the signal. This task is performed using dictionaries designed for sparse reconstruction, optimized to work under the typical noise conditions of gravitational-wave detectors. In the second stage, focused on classification, the enhanced signals are classified by assigning specific labels, whether of astrophysical or instrumental origin. Currently, classification relies on the **LRSDL**¹ model, which leverages the patterns captured by the trained dictionaries to differentiate between various signal classes.

This pipeline was developed as part of my PhD thesis _Gravitational-wave signal denoising, reconstruction and classification via sparse dictionary learning_ (2025).









# References
1. T. H. Vu and V. Monga, _Fast Low-Rank Shared Dictionary Learning for Image Classification_, in IEEE Transactions on Image Processing, vol. 26, no. 11, pp. 5160-5175, Nov. 2017, doi: 10.1109/TIP.2017.2729885.









# TODO

- The training function of LRSDL is from an older version that "expands" the training dataset by obtaining multiple windows from each signal using the usual sliding window. This needs to be reviewed to determine whether it is positive, negative, or unnecessary. In any case, since this step in the dictionary is exclusively about increasing the samples, it should either be removed or placed in a separate function.
- Vectorize `snr()` to estimate multiple signals (rfft accepts multidimensional arrays).
- Make it explicit when iterating over labels by updating the relevant variables: avoid referring to different dictionaries and instead refer to different labels.
- Ensure that `DictionarySpams.reconstruct` and `..._reconstruct` are explicitly designed to reconstruct only a single signal.
- Modularize `extract_patches` function.









# Notes

Below are some notes about different parts of the code.

## Training the Denoising dictionary with SPAMS

### Main training models

In the `spams.trainDL` function:

- **`mode=0`** and **`mode=1`** both formulate dictionary learning as optimization problems, but they differ in **how the sparsity constraint on** $ \alpha_i $ (the sparse representations of the data $ x_i $ in terms of dictionary $ D $) is applied:

**Key Differences Between `mode=0` and `mode=1`**

1. **Objective Function Structure**:
   - **`mode=0`** minimizes the **reconstruction error** (the squared Euclidean distance between $ x_i $ and $ D \alpha_i $), with an **L1-norm constraint on** $ \alpha_i $.
     $$
     \min_{D \in C} \frac{1}{n} \sum_{i=1}^n \frac{1}{2} \| x_i - D \alpha_i \|_2^2 \quad \text{s.t. } \| \alpha_i \|_1 \leq \lambda_1
     $$
   - **`mode=1`** minimizes the **L1-norm of** $ \alpha_i $ directly, subject to a constraint on the **reconstruction error**.
     $$
     \min_{D \in C} \frac{1}{n} \sum_{i=1}^n \| \alpha_i \|_1 \quad \text{s.t. } \| x_i - D \alpha_i \|_2^2 \leq \lambda_1
     $$

2. **Interpretation of the Constraint**:
   - **In `mode=0`**, the L1 constraint ($ \| \alpha_i \|_1 \leq \lambda_1 $) **limits the sparsity of** $ \alpha_i $ directly by restricting the sum of absolute values of its entries. The goal is to minimize the reconstruction error, subject to a sparsity constraint on $ \alpha_i $.
   - **In `mode=1`**, the **reconstruction error is constrained** ($ \| x_i - D \alpha_i \|_2^2 \leq \lambda_1 $), while the objective is to minimize the L1-norm of $ \alpha_i $. Here, `lambda1` controls the allowed reconstruction error, and the algorithm seeks the sparsest representation that meets this accuracy threshold.

3. **Effect on Sparsity and Reconstruction**:
   - **`mode=0`** typically yields representations that aim for **good reconstruction quality**, limited by the sparsity of $ \alpha_i $. The solution tries to fit the data within the allowed sparsity level.
   - **`mode=1`** typically yields **sparser representations** at the expense of some flexibility in reconstruction error. Here, the solution prioritizes minimizing $ \| \alpha_i \|_1 $ while only requiring that the reconstruction error stay within a set limit, leading to fewer non-zero elements in $ \alpha_i $ if this constraint can be met.

**Why Use `mode=1`?**

If your primary goal is to **obtain sparse representations**, even if it means a small sacrifice in reconstruction quality, `mode=1` is appropriate. By minimizing $ \| \alpha_i \|_1 $, the algorithm finds the most compact (sparse) representation for each $ x_i $ that still reconstructs it within the desired error limit.
  
**Why Use `mode=0`?**

If **reconstruction accuracy** is the main focus, with sparsity as a secondary goal, `mode=0` may be better. It minimizes reconstruction error directly, only applying the L1 constraint on $ \alpha_i $ as a limit on how sparse the representation can be.


### Mini-batch relevance and `batch_size` parameter

#### General overview

While in-memory processing allows the dataset to fit comfortably in memory, using mini-batches instead of a single large batch introduces beneficial stochasticity, improves parallel efficiency, and offers more flexibility in convergence control. Setting a moderate `batch_size` (like 256 or 512) even when data is in memory can be advantageous in terms of both convergence quality and computational efficiency.

1. **Stochasticity Benefits**:
   - Mini-batch updates introduce a form of stochasticity that can help the algorithm escape local minima or saddle points, especially in non-convex problems like dictionary learning. By updating the dictionary in smaller steps with each batch, the training process captures diverse data representations across mini-batches, often leading to better generalization and robustness.

2. **Smoother Convergence**:
   - Large batch sizes can make the optimization process more deterministic and potentially more sensitive to specific data patterns in the batch, leading to less smooth convergence. Smaller batches create updates that average over different parts of the data distribution, smoothing out the learning trajectory and often resulting in faster convergence to a stable solution.

3. **Efficient Computation on Multi-Core Systems**:
   - Mini-batches work well with parallelization because they allow each CPU core to process a different batch simultaneously. With a single large batch, all cores might need to synchronize for each update, reducing the efficiency of parallel processing.
   - By splitting the data into smaller batches, SPAMS can leverage multiple cores effectively, reducing computation time per update and improving overall runtime performance.

4. **Incremental Updates with Potential for Early Stopping**:
   - Smaller batch sizes allow more frequent updates to the dictionary, providing opportunities for earlier stopping if convergence criteria are met. With a single large batch, the dictionary is updated less frequently, which could result in slower effective convergence in practice, especially if the dictionary is close to optimal.

#### The case of Gravitational Wave signals

In general, a **lower `batch_size`** is beneficial for learning from complex signals with rich information across a wide range of frequencies and high temporal variability:

1. **Enhanced Stochastic Influence**
   
   Lower batch sizes introduce randomness into the learning process, which is especially helpful for complex, nonstationary signals. This stochastic influence allows the dictionary to adapt to a wider variety of patterns, making it less likely to overfit specific structures and more likely to capture general features of the data.

2. **Better Representation of Diverse Signal Characteristics**
   
   When signals contain intricate variations, each small batch offers a unique sampling of these characteristics. This diversity helps the dictionary learn atoms that are representative of the full signal spectrum, rather than narrowly fitting to specific sections of the data.

3. **Avoiding the Loss of Localized Information**
   
   A large batch size can average out subtle but meaningful variations within the batch, especially if there are transient or time-localized features.

4. **Improved Convergence for High-Dimensional, Complex Patterns**

   Smaller batch sizes allow the dictionary to make frequent, incremental adjustments, which are useful for complex signals where different parts of the data might push updates in diverse directions. This approach smooths the convergence process, allowing the dictionary to gradually settle into a representation that captures the full range of signal variations.

#### Experimental observations

For GW signals, all split into fitting windows and normalized to their L2 norm, the denoising dictionary performed the best **for later classification** when trained with `batch_size = 1` for a large number of iterations, `n_iter = 100_000`, to compensate for the small batch size by allowing the dictionary ample time to adjust and converge.

This configuration essentially maximized stochastic influence by treating each training window as a unique, incremental update to the dictionary—like a continuous learning process, where the dictionary atoms evolve to reflect both common and rare patterns in the data.

It must be mentioned that this comes with the cost of increased computational time, not only due to the high number of iterations needed, but also because parallelization is not viable.